"""
IR Spectroscopy Module for MARS.

Compute IR spectra using analytical Hessian and partial charges.
Supports any registered potential; partial-charge-dependent quantities
(IR intensities) require a potential that provides charges (e.g. SO3LR).
"""

import re
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from .auto_config import EV_TO_KCALMOL
from .log import log_error, log_header, log_info, log_message, log_warning
from .potentials import get_potential
from .utils import (
    KB_EV_PER_K,
    _is_jax_oom,
    check_charge,
    detect_bonds,
    enable_float64,
    get_atomic_masses,
    load_structure,
)


def _select_conformer(structures, conformer_index):
    """Resolve and validate ``conformer_index`` against a list of loaded structures.

    Returns ``(structure, conformer_index)``. Logs the selection.
    """
    n_conformers = len(structures)
    if n_conformers > 1:
        log_info(f"Found {n_conformers} conformers in XYZ file")
        if conformer_index is None:
            conformer_index = 0
            log_info(
                "Using first conformer (index 0). Use --conformer N to select a different one."
            )
        elif conformer_index >= n_conformers:
            log_info(
                f"Warning: conformer index {conformer_index} out of range. Using last conformer."
            )
            conformer_index = n_conformers - 1
        else:
            log_info(f"Using conformer {conformer_index}")
    else:
        conformer_index = 0
    return structures[conformer_index], conformer_index


def _resolve_dtype(use_float64, force_float64_for_md=False):
    """Configure JAX precision and return the matching ``jnp`` dtype.

    For Hessian-based IR, float32 is allowed but float64 is recommended.
    For MD-based IR (``force_float64_for_md=True``), float64 is enforced.
    """
    if use_float64 or force_float64_for_md:
        log_info("Using float64 precision")
        enable_float64()
        return jnp.float64
    log_info("Using float32 precision")
    log_info("For analytical Hessian based IR float64 is recommended (use --float64)")
    return jnp.float32


def _normalize_atom_indices(atom_indices, n_atoms):
    """Validate and canonicalize a user-supplied atom-selection.

    Returns ``None`` when *atom_indices* is ``None`` or selects every
    atom (so the partial-Hessian path is bypassed and the full-system
    routine is used). Otherwise returns a sorted, deduplicated list of
    ``int`` indices in ``[0, n_atoms)``.

    Raises:
        ValueError: if any index is out of range.
    """
    if atom_indices is None:
        return None
    idxs = sorted({int(i) for i in atom_indices})
    if not idxs:
        return None
    if idxs[0] < 0 or idxs[-1] >= n_atoms:
        raise ValueError(
            f"atom_indices out of range: got {idxs[0]}..{idxs[-1]} for a system "
            f"with {n_atoms} atoms (valid range 0..{n_atoms - 1})"
        )
    if len(idxs) == n_atoms and idxs == list(range(n_atoms)):
        return None
    return idxs


def _build_ir_potential(
    potential_name,
    numbers,
    lr_cutoff,
    charge,
    model_path,
    dtype,
    compute_charges=False,
    potential_options=None,
):
    """Build a potential for IR calculations.

    Shared helper used by ``compute_ir_spectrum`` and ``compute_ir_from_md``.
    ``potential_options`` is an optional dict that forwards potential-specific
    arguments such as ``mace_foundation``, ``mace_model`` or ``dxtb_method``.

    Returns:
        potential: The constructed PotentialWrapper instance.
    """
    potential_kwargs = {}
    options = potential_options or {}

    if potential_name == "so3lr":
        potential_kwargs.update(
            species=numbers,
            lr_cutoff=lr_cutoff,
            charge=charge,
            model=model_path,
            dtype=dtype,
        )
        if compute_charges:
            potential_kwargs["compute_charges"] = True
    elif potential_name == "mace":
        potential_kwargs.update(
            species=numbers,
            dtype=dtype,
            charge=charge,
        )
        if options.get("mace_foundation"):
            potential_kwargs["foundation"] = options["mace_foundation"]
        mace_model = options.get("mace_model") or model_path
        if mace_model:
            potential_kwargs["model"] = mace_model
        if options.get("mace_cache_dir"):
            potential_kwargs["cache_dir"] = options["mace_cache_dir"]
        if options.get("mace_spin") is not None:
            potential_kwargs["spin"] = options["mace_spin"]
    elif potential_name == "dxtb":
        potential_kwargs.update(
            species=numbers,
            charge=charge,
            dtype=dtype,
        )
        if options.get("dxtb_method"):
            potential_kwargs["method"] = options["dxtb_method"]

    try:
        check_charge({"numbers": numbers}, charge=charge)
    except ValueError as exc:
        log_error(str(exc))
        sys.exit(1)
    try:
        potential = get_potential(potential_name, **potential_kwargs)
    except ValueError:
        log_error(f"Unknown potential '{potential_name}'")
        sys.exit(1)
    return potential


# ============================================================================
# Physical Constants
# ============================================================================

# Conversion factors
AMU_TO_KG = 1.66053906660e-27  # amu to kg
ANGSTROM_TO_M = 1e-10  # Angstrom to meters
EV_TO_J = 1.602176634e-19  # eV to Joules
C_LIGHT = 299792458.0  # speed of light in m/s
C_LIGHT_CMS = C_LIGHT * 100.0  # speed of light in cm/s
HBAR = 1.054571817e-34  # reduced Planck constant in J·s
KB_J = 1.380649e-23  # Boltzmann constant in J/K
CM_PER_PS = C_LIGHT * 1e-12 / ANGSTROM_TO_M  # cm/ps

# For wavenumber conversion: ω (rad/s) = 2πc·ν (cm⁻¹)
# ν (cm⁻¹) = ω / (2πc)
FREQ_TO_WAVENUMBER = 1.0 / (2.0 * np.pi * C_LIGHT * 100.0)  # rad/s to cm⁻¹


# ============================================================================
# Hessian Computation
# ============================================================================


def compute_hessian(
    energy_fn,
    positions: jnp.ndarray,
    use_jit: bool = True,
    atom_indices: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Compute analytical Hessian matrix using JAX automatic differentiation.

    The Hessian is the matrix of second derivatives of energy with respect
    to atomic positions:
        H[3i+α, 3j+β] = ∂²E / ∂r_iα ∂r_jβ

    Args:
        energy_fn: JAX energy function that takes positions (n_atoms, 3)
        positions: (n_atoms, 3) atomic positions in Angstrom
        use_jit: If True, JIT-compile the Hessian computation for speed
        atom_indices: If given, compute a partial Hessian restricted to
            this 0-based subset of atoms. The energy still uses the full
            system; only the selected atoms' displacements are propagated.
            Returns a (3K, 3K) matrix where K = len(atom_indices).

    Returns:
        hessian: (3*n_atoms, 3*n_atoms) Hessian matrix in eV/Angstrom²,
            or (3*K, 3*K) when ``atom_indices`` is given.
    """
    hessian, _ = compute_hessian_and_dipole_derivs(
        energy_fn, None, positions, use_jit=use_jit, atom_indices=atom_indices
    )
    return hessian


def compute_hessian_and_dipole_derivs(
    energy_fn, charges_fn, positions, use_jit=True, atom_indices=None
):
    """Compute (partial) Hessian and dipole derivatives in a single pass.

    When ``atom_indices`` is provided, only those atoms' Cartesian
    displacements drive the differentiation: the rest of the system stays
    fixed at *positions*. The energy and (full-system) dipole still see
    every atom — this is the standard partial-Hessian construction.

    Returns:
        hessian: (3*n_atoms, 3*n_atoms) or (3K, 3K) Hessian.
        dipole_derivs: (3*n_atoms, 3) or (3K, 3) dipole derivatives, or
            ``None`` when *charges_fn* is ``None``.
    """
    n_atoms = positions.shape[0]

    if atom_indices is None:
        pos_flat = positions.flatten()
        if charges_fn is not None:

            def grad_and_dipole(pf):
                pos = pf.reshape(n_atoms, 3)
                g = jax.grad(lambda p: energy_fn(p.reshape(n_atoms, 3)))(pf)
                d = jnp.sum(charges_fn(pos)[:, None] * pos, axis=0)
                return jnp.concatenate([g, d])

            jac_fn = (
                jax.jit(jax.jacfwd(grad_and_dipole)) if use_jit else jax.jacfwd(grad_and_dipole)
            )
            jac = jac_fn(pos_flat)
            return jac[: 3 * n_atoms, :], jac[3 * n_atoms :, :].T
        ef = lambda pf: energy_fn(pf.reshape(n_atoms, 3))
        hfn = jax.jit(jax.hessian(ef)) if use_jit else jax.hessian(ef)
        return hfn(pos_flat), None

    # Partial Hessian: differentiate only with respect to the selected
    # atoms' Cartesian coordinates. The non-selected atoms stay at
    # ``positions`` so the energy still sees the whole system.
    atom_idx_arr = jnp.asarray(atom_indices, dtype=jnp.int32)
    k = int(atom_idx_arr.shape[0])
    n_dof_sel = 3 * k
    base_positions = positions
    x_sel_flat = base_positions[atom_idx_arr].reshape(-1)

    def _scatter_full(x_sel):
        return base_positions.at[atom_idx_arr].set(x_sel.reshape(k, 3))

    def energy_sel(x_sel):
        return energy_fn(_scatter_full(x_sel))

    if charges_fn is not None:

        def grad_and_dipole_sel(x_sel):
            g = jax.grad(energy_sel)(x_sel)
            full_pos = _scatter_full(x_sel)
            d = jnp.sum(charges_fn(full_pos)[:, None] * full_pos, axis=0)
            return jnp.concatenate([g, d])

        jac_fn = (
            jax.jit(jax.jacfwd(grad_and_dipole_sel)) if use_jit else jax.jacfwd(grad_and_dipole_sel)
        )
        jac = jac_fn(x_sel_flat)
        return jac[:n_dof_sel, :], jac[n_dof_sel:, :].T

    hfn = jax.jit(jax.hessian(energy_sel)) if use_jit else jax.hessian(energy_sel)
    return hfn(x_sel_flat), None


def compute_hessian_fd(
    energy_fn,
    positions: jnp.ndarray,
    displacement: float = 0.01,
    atom_indices: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Compute Hessian matrix by finite differences of the gradient.

    Uses central differences on the gradient (forces):
        H[3i+α, 3j+β] ≈ (∂E/∂r_iα(+δ_jβ) - ∂E/∂r_iα(-δ_jβ)) / (2δ)

    Args:
        energy_fn: JAX energy function that takes positions (n_atoms, 3)
        positions: (n_atoms, 3) atomic positions in Angstrom
        displacement: Finite displacement step in Angstrom (default: 0.01)
        atom_indices: If given, only displace these 0-based atom indices
            and slice the gradient to their components — yielding a
            (3K, 3K) partial Hessian. Gradients are still evaluated on
            the full system (all forces, partial displacements).

    Returns:
        hessian: (3*n_atoms, 3*n_atoms) or (3K, 3K) Hessian matrix in eV/Å²
    """
    hessian, _ = compute_hessian_and_dipole_derivs_fd(
        energy_fn, None, positions, displacement=displacement, atom_indices=atom_indices
    )
    return hessian


def compute_hessian_and_dipole_derivs_fd(
    energy_fn,
    charges_fn,
    positions: jnp.ndarray,
    displacement: float = 0.01,
    atom_indices: Optional[List[int]] = None,
) -> Tuple[jnp.ndarray, Optional[np.ndarray]]:
    """Compute Hessian and dipole derivatives together in a single FD pass.

    Both quantities require the same ±displaced geometries, so computing them
    together costs no extra energy/gradient evaluations compared with calling
    the two functions separately.

    Uses central differences:
        H[:, j]     ≈ (grad(+δ_j) − grad(−δ_j)) / (2δ)
        dμ/dr[j, :] ≈ (μ(+δ_j)   − μ(−δ_j))   / (2δ)

    Args:
        energy_fn: JAX energy function ``(n_atoms, 3) → scalar``.
        charges_fn: Function ``(n_atoms, 3) → (n_atoms,)`` or ``None``.
                    When ``None``, dipole derivatives are not computed.
        positions: ``(n_atoms, 3)`` atomic positions in Angstrom.
        displacement: Step size in Angstrom (default: 0.01).
        atom_indices: If given, only displace the selected atoms.
            Gradients are computed on the full system (so non-selected
            atoms still contribute their forces), but the returned
            Hessian is the (3K, 3K) sub-block restricted to selected
            atoms — the standard partial-Hessian construction.

    Returns:
        hessian: ``(3*n_atoms, 3*n_atoms)`` or ``(3K, 3K)`` Hessian in eV/Å².
        dipole_derivs: ``(3*n_atoms, 3)`` or ``(3K, 3)`` dipole derivatives
                       in e·Å/Å, or ``None`` when *charges_fn* is ``None``.
    """
    n_atoms = positions.shape[0]
    pos_flat = np.array(positions).flatten()

    if atom_indices is None:
        sel_atoms = np.arange(n_atoms, dtype=int)
    else:
        sel_atoms = np.asarray(atom_indices, dtype=int)

    sel_dof = np.array([3 * a + d for a in sel_atoms for d in range(3)], dtype=int)
    n_dof_sel = sel_dof.size

    grad_fn = jax.jit(jax.grad(lambda p: energy_fn(p.reshape(n_atoms, 3))))

    hessian = np.zeros((n_dof_sel, n_dof_sel))
    dipole_derivs = np.zeros((n_dof_sel, 3)) if charges_fn is not None else None

    for j_local in range(n_dof_sel):
        j_global = int(sel_dof[j_local])

        pos_plus = pos_flat.copy()
        pos_minus = pos_flat.copy()
        pos_plus[j_global] += displacement
        pos_minus[j_global] -= displacement

        p_plus_jax = jnp.array(pos_plus.reshape(n_atoms, 3))
        p_minus_jax = jnp.array(pos_minus.reshape(n_atoms, 3))

        grad_plus = np.array(grad_fn(p_plus_jax.flatten()))
        grad_minus = np.array(grad_fn(p_minus_jax.flatten()))
        hessian[:, j_local] = (grad_plus[sel_dof] - grad_minus[sel_dof]) / (2.0 * displacement)

        if charges_fn is not None:
            q_plus = np.array(charges_fn(p_plus_jax))
            q_minus = np.array(charges_fn(p_minus_jax))
            mu_plus = np.sum(q_plus[:, None] * pos_plus.reshape(n_atoms, 3), axis=0)
            mu_minus = np.sum(q_minus[:, None] * pos_minus.reshape(n_atoms, 3), axis=0)
            dipole_derivs[j_local] = (mu_plus - mu_minus) / (2.0 * displacement)

    hessian = 0.5 * (hessian + hessian.T)
    return jnp.array(hessian), dipole_derivs


def mass_weight_hessian(hessian: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """Mass-weight the Hessian matrix.

    Mass-weighted Hessian: H_mw[3i+α, 3j+β] = H[3i+α, 3j+β] / sqrt(m_i * m_j)

    Args:
        hessian: (3*n_atoms, 3*n_atoms) Hessian in eV/Angstrom²
        masses: (n_atoms,) atomic masses in amu

    Returns:
        hessian_mw: (3*n_atoms, 3*n_atoms) mass-weighted Hessian
    """
    # Create mass weighting matrix: 1/sqrt(m_i * m_j) for each 3x3 block
    mass_matrix = jnp.repeat(masses, 3)  # (3*n_atoms,)
    mass_weight = 1.0 / jnp.sqrt(jnp.outer(mass_matrix, mass_matrix))

    # Apply mass weighting
    hessian_mw = hessian * mass_weight

    return hessian_mw


# ============================================================================
# Normal Mode Analysis
# ============================================================================


def compute_normal_modes(
    hessian_mw: jnp.ndarray, masses: jnp.ndarray, temperature: float = 300.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute vibrational normal modes from mass-weighted Hessian.

    Solves the eigenvalue problem: H_mw · L = ω² · L
    where L are the normal mode eigenvectors and ω² are the squared frequencies.

    Args:
        hessian_mw: (3*n_atoms, 3*n_atoms) mass-weighted Hessian
        masses: (n_atoms,) atomic masses in amu
        temperature: Temperature in Kelvin for thermal population calculation

    Returns:
        frequencies: (3*n_atoms,) vibrational frequencies in cm⁻¹
        normal_modes: (3*n_atoms, 3*n_atoms) normal mode eigenvectors (columns)
        thermal_populations: (3*n_atoms,) Boltzmann populations at given temperature
    """

    # Diagonalize mass-weighted Hessian
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian_mw)

    # Convert eigenvalues to frequencies
    # ω² = eigenvalue → ω = sqrt(eigenvalue) if positive, imaginary if negative
    # Units: eigenvalue in eV/Angstrom²/amu

    # Conversion factor from internal units to rad/s:
    # [eV/Angstrom²/amu] → [J/m²/kg] = [kg/s²]
    # ω = sqrt(eigenvalue * conversion_factor)
    conversion = EV_TO_J / (ANGSTROM_TO_M**2 * AMU_TO_KG)

    # Compute angular frequencies in rad/s
    omega_sq = eigenvalues * conversion
    omega = jnp.sqrt(jnp.abs(omega_sq)) * jnp.sign(omega_sq)

    # Convert to wavenumbers (cm⁻¹)
    frequencies = omega * FREQ_TO_WAVENUMBER

    # Compute thermal populations using Boltzmann distribution
    # For harmonic oscillator: P_n ∝ exp(-n·ℏω / kT)
    # Ground state population: P_0 = 1 / (1 + exp(-ℏω/kT))
    kT = temperature * KB_EV_PER_K * EV_TO_J  # in Joules

    # Avoid division by zero for very small frequencies
    omega_safe = jnp.where(jnp.abs(omega) > 1e-6, omega, 1e-6)
    exponent = HBAR * jnp.abs(omega_safe) / kT
    thermal_populations = 1.0 / (1.0 - jnp.exp(-exponent))

    # Set population to 1 for translation/rotation modes (near-zero frequencies)
    thermal_populations = jnp.where(jnp.abs(frequencies) < 50.0, 1.0, thermal_populations)

    return frequencies, eigenvectors, thermal_populations


# ============================================================================
# Dipole and IR Intensity
# ============================================================================


def compute_dipole_derivatives(
    charges_fn: Callable,
    positions: jnp.ndarray,
    atom_indices: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Compute derivatives of dipole moment with respect to atomic positions.

    The dipole derivative determines IR intensity:
        ∂μ/∂r_iα where μ is the dipole moment

    For a system with partial charges, the dipole is:
        μ = Σ_i q_i · r_i

    Args:
        charges_fn: Function that returns charges given positions
        positions: (n_atoms, 3) atomic positions in Angstrom
        atom_indices: If given, only derive with respect to these 0-based
            atom indices. The dipole sum still runs over all atoms.

    Returns:
        dipole_derivatives: (3*n_atoms, 3) or (3K, 3) derivatives in
        e·Angstrom/Angstrom.
    """
    n_atoms = positions.shape[0]

    if atom_indices is None:

        def dipole_moment(pos):
            return jnp.sum(charges_fn(pos)[:, None] * pos, axis=0)

        jacobian_fn = jax.jacrev(dipole_moment)
        dipole_derivs = jacobian_fn(positions)  # (3, n_atoms, 3)
        return dipole_derivs.transpose(1, 2, 0).reshape(3 * n_atoms, 3)

    atom_idx_arr = jnp.asarray(atom_indices, dtype=jnp.int32)
    k = int(atom_idx_arr.shape[0])
    base_positions = positions
    x_sel_flat = base_positions[atom_idx_arr].reshape(-1)

    def dipole_moment_sel(x_sel):
        full_pos = base_positions.at[atom_idx_arr].set(x_sel.reshape(k, 3))
        return jnp.sum(charges_fn(full_pos)[:, None] * full_pos, axis=0)

    jacobian_fn = jax.jacrev(dipole_moment_sel)
    dipole_derivs = jacobian_fn(x_sel_flat)  # (3, 3K)
    return dipole_derivs.T


def compute_ir_intensities(
    normal_modes: jnp.ndarray, dipole_derivatives: jnp.ndarray, masses: jnp.ndarray
) -> jnp.ndarray:
    """Compute IR intensities for each normal mode.

    IR intensity is proportional to |∂μ/∂Q|² where Q is the normal coordinate.

    The transformation from Cartesian to normal coordinates is:
        ∂μ/∂Q_k = Σ_iα (∂μ/∂r_iα) · (∂r_iα/∂Q_k)
                = Σ_iα (∂μ/∂r_iα) · L_k[iα] / sqrt(m_i)

    Args:
        normal_modes: (3*n_atoms, 3*n_atoms) normal mode eigenvectors (columns)
        dipole_derivatives: (3*n_atoms, 3) dipole derivatives in e·Angstrom/Angstrom
        masses: (n_atoms,) atomic masses in amu

    Returns:
        intensities: (3*n_atoms,) IR intensities in arbitrary units (proportional to km/mol)
    """
    # Un-mass-weight the normal modes
    # L_k[iα] / sqrt(m_i) for transformation to normal coordinates
    mass_inv_sqrt = 1.0 / jnp.sqrt(jnp.repeat(masses, 3))
    normal_modes_unweighted = normal_modes * mass_inv_sqrt[:, None]

    # Compute dipole derivatives in normal coordinates
    # ∂μ/∂Q_k = Σ_iα (∂μ/∂r_iα) · L_k[iα] / sqrt(m_i)
    # Shape: (3, n_modes)
    dipole_deriv_Q = dipole_derivatives.T @ normal_modes_unweighted

    # IR intensity proportional to |∂μ/∂Q|²
    intensities = jnp.sum(dipole_deriv_Q**2, axis=0)

    return intensities


# ============================================================================
# Vibrational Mode Classification
# ============================================================================

# Color map for mode types (used in plotting). Defined in a JAX-free module so
# the lightweight viewer can import it without pulling in the potential stack.
from .spectro_constants import MODE_TYPE_COLORS  # noqa: E402,F401

# Bond order symbols for labels
BOND_ORDER_SYMBOLS = {1: "-", 2: "=", 3: "#"}

# Typical valences for common elements (most stable neutral valence).
# For elements with multiple common valences, the most common one is listed first.
TYPICAL_VALENCES = {
    1: [1],  # H
    5: [3],  # B
    6: [4],  # C
    7: [3, 5],  # N
    8: [2],  # O
    9: [1],  # F
    14: [4],  # Si
    15: [3, 5],  # P
    16: [2, 4, 6],  # S
    17: [1],  # Cl
    35: [1],  # Br
    53: [1],  # I
}


def _get_target_valence(z: int, current_bonds: int) -> int:
    """Return the target valence for an atom given its atomic number.

    Picks the smallest typical valence that is >= current_bonds.
    Falls back to current_bonds if the element is unknown.

    Args:
        z: Atomic number
        current_bonds: Number of bonds already assigned to this atom

    Returns:
        Target valence (total bond order sum the atom should have)
    """
    valences = TYPICAL_VALENCES.get(z)
    if valences is None:
        return current_bonds
    for v in valences:
        if v >= current_bonds:
            return v
    return valences[-1]


def assign_bond_orders(bonds: list, atomic_numbers: np.ndarray, charge: int = 0) -> dict:
    """Assign bond orders using valence rules on the molecular connectivity graph.

    Algorithm:
        1. Start with all bonds as single (order = 1).
        2. Compute each atom's valence deficit: target_valence - sum_of_bond_orders.
        3. Iteratively promote bonds where both endpoint atoms have a positive
           deficit, choosing the bond with the largest combined deficit first.
        4. Stop when no more promotions are possible or all deficits are zero.

    The total molecular charge is accounted for by adjusting the available
    electron count, which limits the total number of bond-order increases.

    Args:
        bonds: List of (i, j) bond tuples
        atomic_numbers: (n_atoms,) atomic numbers
        charge: Total molecular charge (default: 0)

    Returns:
        Dictionary mapping (i, j) -> bond_order for each bond (with i < j)
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    n_atoms = len(atomic_numbers)

    # Initialize all bonds as single
    bond_orders = {}
    for i, j in bonds:
        key = (min(i, j), max(i, j))
        bond_orders[key] = 1

    # Build adjacency: atom -> list of bond keys
    atom_bonds = {a: [] for a in range(n_atoms)}
    for key in bond_orders:
        i, j = key
        atom_bonds[i].append(key)
        atom_bonds[j].append(key)

    def current_bond_order_sum(atom):
        return sum(bond_orders[bk] for bk in atom_bonds[atom])

    # Total electrons available for bonding (rough estimate)
    total_valence_electrons = sum(
        _get_target_valence(int(z), len(atom_bonds[a])) for a, z in enumerate(atomic_numbers)
    )
    # Each bond order increase of 1 uses 2 electrons (one shared pair)
    # Current single bonds already use 2*n_bonds electrons
    max_extra_pairs = (total_valence_electrons - 2 * len(bonds) - charge) // 2
    extra_pairs_used = 0

    # Iteratively promote bonds
    changed = True
    while changed:
        changed = False

        # Compute deficits
        deficits = np.zeros(n_atoms, dtype=int)
        for a in range(n_atoms):
            target = _get_target_valence(int(atomic_numbers[a]), len(atom_bonds[a]))
            deficits[a] = target - current_bond_order_sum(a)

        # Collect candidate bonds for promotion (both atoms have deficit > 0,
        # bond order < 3)
        candidates = []
        for key in bond_orders:
            i, j = key
            if bond_orders[key] < 3 and deficits[i] > 0 and deficits[j] > 0:
                combined = deficits[i] + deficits[j]
                candidates.append((combined, key))

        if not candidates:
            break

        # Sort by combined deficit (highest first) for greedy assignment
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, key in candidates:
            i, j = key
            # Re-check deficits (they may have changed from earlier promotions)
            target_i = _get_target_valence(int(atomic_numbers[i]), len(atom_bonds[i]))
            target_j = _get_target_valence(int(atomic_numbers[j]), len(atom_bonds[j]))
            deficit_i = target_i - current_bond_order_sum(i)
            deficit_j = target_j - current_bond_order_sum(j)

            if (
                deficit_i > 0
                and deficit_j > 0
                and bond_orders[key] < 3
                and extra_pairs_used < max_extra_pairs
            ):
                bond_orders[key] += 1
                extra_pairs_used += 1
                changed = True

    return bond_orders


def _build_topology(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    symbols: list,
    tolerance: float = 1.2,
    charge: int = 0,
) -> dict:
    """Build molecular topology: bonds, angles, and dihedrals.

    Args:
        positions: (n_atoms, 3) atomic positions in Angstrom
        atomic_numbers: (n_atoms,) atomic numbers
        symbols: List of element symbols
        tolerance: Bond detection tolerance factor (default: 1.2)
        charge: Total molecular charge for bond order assignment

    Returns:
        Dictionary with bonds, angles, dihedrals, neighbors, bond_labels,
        bond_orders
    """
    bonds, bond_distances = detect_bonds(positions, atomic_numbers, tolerance=tolerance)

    # Assign bond orders using valence rules
    bond_orders = assign_bond_orders(bonds, atomic_numbers, charge=charge)

    # Build adjacency list
    n_atoms = len(positions)
    neighbors = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    # Build bond labels with bond order symbols: "C=O", "C#N", "C-H" etc.
    bond_labels = {}
    for i, j in bonds:
        key = (min(i, j), max(i, j))
        order = bond_orders.get(key, 1)
        sym = BOND_ORDER_SYMBOLS.get(order, "-")
        bond_labels[(i, j)] = f"{symbols[i]}{sym}{symbols[j]}"
        bond_labels[(j, i)] = f"{symbols[j]}{sym}{symbols[i]}"

    # Generate angles: (i, j, k) where j is central atom
    angles = []
    for j in range(n_atoms):
        nbrs = sorted(neighbors[j])
        for idx_a in range(len(nbrs)):
            for idx_b in range(idx_a + 1, len(nbrs)):
                i, k = nbrs[idx_a], nbrs[idx_b]
                angles.append((i, j, k))

    # Generate dihedrals: (i, j, k, l)
    dihedrals = []
    for j, k in bonds:
        for i in neighbors[j]:
            if i == k:
                continue
            for l in neighbors[k]:
                if l == j or l == i:
                    continue
                dihedrals.append((i, j, k, l))

    return {
        "bonds": bonds,
        "bond_distances": bond_distances,
        "bond_orders": bond_orders,
        "angles": angles,
        "dihedrals": dihedrals,
        "neighbors": neighbors,
        "bond_labels": bond_labels,
    }


def _mode_to_cartesian_displacements(normal_modes: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Convert mass-weighted eigenvectors to normalized Cartesian displacements.

    Args:
        normal_modes: (3N, 3N) mass-weighted eigenvectors (columns)
        masses: (n_atoms,) atomic masses in amu

    Returns:
        (3N, 3N) normalized Cartesian displacement vectors (columns)
    """
    mass_inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    cartesian_modes = normal_modes * mass_inv_sqrt[:, None]

    # Normalize each mode column
    for k in range(cartesian_modes.shape[1]):
        norm = np.linalg.norm(cartesian_modes[:, k])
        if norm > 1e-10:
            cartesian_modes[:, k] /= norm

    return cartesian_modes


def _project_mode_onto_stretches(
    displacement: np.ndarray, positions: np.ndarray, bonds: list
) -> Tuple[float, list]:
    """Project a mode onto bond-stretching internal coordinates.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement for one mode
        positions: (n_atoms, 3) equilibrium positions
        bonds: List of (i, j) bond tuples

    Returns:
        total_score: Sum of squared stretching projections
        per_bond: List of (bond, score) for each bond
    """
    per_bond = []
    total = 0.0

    for i, j in bonds:
        r_ij = positions[j] - positions[i]
        r_norm = np.linalg.norm(r_ij)
        if r_norm < 1e-10:
            per_bond.append(((i, j), 0.0))
            continue
        e_ij = r_ij / r_norm

        # Relative displacement along bond direction
        delta = np.dot(displacement[j] - displacement[i], e_ij)
        score = delta**2
        per_bond.append(((i, j), score))
        total += score

    return total, per_bond


def _project_mode_onto_bends(
    displacement: np.ndarray, positions: np.ndarray, angles: list
) -> Tuple[float, list]:
    """Project a mode onto angle-bending internal coordinates using Wilson B-matrix.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement for one mode
        positions: (n_atoms, 3) equilibrium positions
        angles: List of (i, j, k) angle tuples (j is central)

    Returns:
        total_score: Sum of squared bending projections
        per_angle: List of (angle, score) for each angle
    """
    per_angle = []
    total = 0.0

    for i, j, k in angles:
        r_ji = positions[i] - positions[j]
        r_jk = positions[k] - positions[j]
        r_ji_norm = np.linalg.norm(r_ji)
        r_jk_norm = np.linalg.norm(r_jk)

        if r_ji_norm < 1e-10 or r_jk_norm < 1e-10:
            per_angle.append(((i, j, k), 0.0))
            continue

        e_ji = r_ji / r_ji_norm
        e_jk = r_jk / r_jk_norm

        cos_theta = np.clip(np.dot(e_ji, e_jk), -1.0, 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        # Guard against linear angles
        if sin_theta < 1e-6:
            per_angle.append(((i, j, k), 0.0))
            continue

        # Wilson B-matrix elements for bending
        B_i = (cos_theta * e_ji - e_jk) / (r_ji_norm * sin_theta)
        B_k = (cos_theta * e_jk - e_ji) / (r_jk_norm * sin_theta)
        B_j = -(B_i + B_k)

        # Angular change
        d_theta = (
            np.dot(B_i, displacement[i])
            + np.dot(B_j, displacement[j])
            + np.dot(B_k, displacement[k])
        )

        score = d_theta**2
        per_angle.append(((i, j, k), score))
        total += score

    return total, per_angle


def _project_mode_onto_torsions(
    displacement: np.ndarray, positions: np.ndarray, dihedrals: list
) -> Tuple[float, list]:
    """Project a mode onto torsion/dihedral internal coordinates using Wilson B-matrix.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement for one mode
        positions: (n_atoms, 3) equilibrium positions
        dihedrals: List of (i, j, k, l) dihedral tuples

    Returns:
        total_score: Sum of squared torsion projections
        per_dihedral: List of (dihedral, score) for each dihedral
    """
    per_dihedral = []
    total = 0.0

    for i, j, k, l in dihedrals:
        b1 = positions[j] - positions[i]
        b2 = positions[k] - positions[j]
        b3 = positions[l] - positions[k]

        b2_norm = np.linalg.norm(b2)
        if b2_norm < 1e-10:
            per_dihedral.append(((i, j, k, l), 0.0))
            continue

        # Plane normals
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_sq = np.dot(n1, n1)
        n2_sq = np.dot(n2, n2)

        if n1_sq < 1e-20 or n2_sq < 1e-20:
            per_dihedral.append(((i, j, k, l), 0.0))
            continue

        # Wilson B-matrix elements for torsion
        B_i = -b2_norm / n1_sq * n1
        B_l = b2_norm / n2_sq * n2

        # Middle atoms from standard formulas
        b1_dot_b2 = np.dot(b1, b2)
        b3_dot_b2 = np.dot(b3, b2)
        b2_sq = b2_norm**2

        B_j = (b1_dot_b2 / b2_sq - 1.0) * B_i - (b3_dot_b2 / b2_sq) * B_l
        B_k = (b3_dot_b2 / b2_sq - 1.0) * B_l - (b1_dot_b2 / b2_sq) * B_i

        # Dihedral angle change
        d_phi = (
            np.dot(B_i, displacement[i])
            + np.dot(B_j, displacement[j])
            + np.dot(B_k, displacement[k])
            + np.dot(B_l, displacement[l])
        )

        score = d_phi**2
        per_dihedral.append(((i, j, k, l), score))
        total += score

    return total, per_dihedral


def _classify_single_mode(
    stretch_score: float, bend_score: float, torsion_score: float, mixed_threshold: float = 0.6
) -> Tuple[str, dict]:
    """Classify a mode based on its projection scores.

    Args:
        stretch_score: Total stretching projection score
        bend_score: Total bending projection score
        torsion_score: Total torsion projection score
        mixed_threshold: Fraction above which the dominant type is assigned

    Returns:
        (type_string, score_fractions_dict)
    """
    total = stretch_score + bend_score + torsion_score
    if total < 1e-15:
        return "mixed", {"stretching": 0.0, "bending": 0.0, "torsion": 0.0}

    fracs = {
        "stretching": stretch_score / total,
        "bending": bend_score / total,
        "torsion": torsion_score / total,
    }

    dominant = max(fracs, key=fracs.get)
    if fracs[dominant] >= mixed_threshold:
        return dominant, fracs
    else:
        return "mixed", fracs


def _identify_participating_atoms(
    displacement: np.ndarray,
    positions: np.ndarray,
    topology: dict,
    mode_type: str,
    symbols: list,
    per_bond_scores: list,
    per_angle_scores: list,
    per_dihedral_scores: list,
) -> Tuple[list, str]:
    """Identify atoms participating most in a mode and generate a chemical label.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement for one mode
        positions: (n_atoms, 3) equilibrium positions
        topology: Topology dictionary
        mode_type: 'stretching', 'bending', 'torsion', or 'mixed'
        symbols: Element symbols
        per_bond_scores: Per-bond projection scores
        per_angle_scores: Per-angle projection scores
        per_dihedral_scores: Per-dihedral projection scores

    Returns:
        (atom_indices, label_string)
    """
    bond_orders = topology.get("bond_orders", {})

    def _bond_sym(a, b):
        """Get the bond order symbol between atoms a and b."""
        key = (min(a, b), max(a, b))
        order = bond_orders.get(key, 1)
        return BOND_ORDER_SYMBOLS.get(order, "-")

    if mode_type == "stretching" and per_bond_scores:
        best = max(per_bond_scores, key=lambda x: x[1])
        bond, _ = best
        i, j = bond
        sym = _bond_sym(i, j)
        label = f"{symbols[i]}{sym}{symbols[j]} stretch"
        return [i, j], label

    elif mode_type == "bending" and per_angle_scores:
        best = max(per_angle_scores, key=lambda x: x[1])
        angle, _ = best
        i, j, k = angle
        s1, s2 = _bond_sym(i, j), _bond_sym(j, k)
        label = f"{symbols[i]}{s1}{symbols[j]}{s2}{symbols[k]} bend"
        return [i, j, k], label

    elif mode_type == "torsion" and per_dihedral_scores:
        best = max(per_dihedral_scores, key=lambda x: x[1])
        dihedral, _ = best
        i, j, k, l = dihedral
        s1, s2, s3 = _bond_sym(i, j), _bond_sym(j, k), _bond_sym(k, l)
        label = f"{symbols[i]}{s1}{symbols[j]}{s2}{symbols[k]}{s3}{symbols[l]} torsion"
        return [i, j, k, l], label

    else:
        # Mixed or fallback: find atoms with largest displacement
        mag = np.linalg.norm(displacement, axis=1)
        top_idx = np.argsort(mag)[-3:][::-1]

        # Try to identify dominant contribution
        all_scores = []
        if per_bond_scores:
            best_bond = max(per_bond_scores, key=lambda x: x[1])
            all_scores.append(("stretch", best_bond[1], best_bond[0]))
        if per_angle_scores:
            best_angle = max(per_angle_scores, key=lambda x: x[1])
            all_scores.append(("bend", best_angle[1], best_angle[0]))
        if per_dihedral_scores:
            best_dih = max(per_dihedral_scores, key=lambda x: x[1])
            all_scores.append(("torsion", best_dih[1], best_dih[0]))

        if all_scores:
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top = all_scores[0]
            atoms = list(top[2])
            # Build label with bond order symbols between consecutive atoms
            parts = [symbols[atoms[0]]]
            for idx_a in range(1, len(atoms)):
                parts.append(_bond_sym(atoms[idx_a - 1], atoms[idx_a]))
                parts.append(symbols[atoms[idx_a]])
            sym_str = "".join(parts)
            label = f"{sym_str} mixed"
        else:
            label = "unclassified"
            atoms = list(top_idx)

        return list(top_idx), label


def _peak_label(mode_dict: dict) -> str:
    """Return a minimal atom-type label for plot peak annotations.

    Extracts just the bond-type prefix, e.g. 'O-H', 'C=O', 'C-H', discarding
    mode sub-type text and fragment tags so the plot stays readable.
    """
    fg = mode_dict.get("functional_group") or ""
    if fg:
        # "O-H (carboxylic acid)" → "O-H"
        return fg.split(" (")[0].split(" [")[0]
    # Fall back: extract first bond pattern from the mode label
    label = mode_dict.get("label", "")
    m = re.match(r"^([A-Z][a-z]?(?:[=\-][A-Z][a-z]?)+)", label)
    return m.group(1) if m else label[:6]


def _sub_classify_bending(
    displacement: np.ndarray, positions: np.ndarray, dominant_angle: Tuple[int, int, int]
) -> str:
    """Sub-classify a bending mode into scissoring, rocking, wagging, or twisting.

    Analyzes relative phase and direction of terminal atom displacements
    with respect to the plane defined by the angle.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement
        positions: (n_atoms, 3) equilibrium positions
        dominant_angle: (i, j, k) the dominant angle

    Returns:
        Sub-type string: 'scissoring', 'rocking', 'wagging', 'twisting', or ''
    """
    i, j, k = dominant_angle

    r_ji = positions[i] - positions[j]
    r_jk = positions[k] - positions[j]

    r_ji_norm = np.linalg.norm(r_ji)
    r_jk_norm = np.linalg.norm(r_jk)
    if r_ji_norm < 1e-10 or r_jk_norm < 1e-10:
        return ""

    e_ji = r_ji / r_ji_norm
    e_jk = r_jk / r_jk_norm

    # Normal to the i-j-k plane
    normal = np.cross(e_ji, e_jk)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-10:
        return ""  # Linear angle
    normal = normal / n_norm

    # Bisector direction (in-plane)
    bisector = e_ji + e_jk
    b_norm = np.linalg.norm(bisector)
    if b_norm < 1e-10:
        bisector = e_ji
    else:
        bisector = bisector / b_norm

    # Relative displacements of terminal atoms w.r.t. central atom
    d_i = displacement[i] - displacement[j]
    d_k = displacement[k] - displacement[j]

    # Project onto out-of-plane (normal) and in-plane components
    d_i_oop = np.dot(d_i, normal)
    d_k_oop = np.dot(d_k, normal)

    ip_mag = np.linalg.norm(d_i - d_i_oop * normal) + np.linalg.norm(d_k - d_k_oop * normal)
    oop_mag = abs(d_i_oop) + abs(d_k_oop)

    if ip_mag + oop_mag < 1e-10:
        return ""

    if oop_mag > ip_mag:
        # Out-of-plane dominant
        if d_i_oop * d_k_oop > 0:
            return "wagging"  # symmetric out-of-plane deformation
        else:
            return "twisting"  # asymmetric out-of-plane deformation
    else:
        # In-plane dominant
        d_i_ip = d_i - d_i_oop * normal
        d_k_ip = d_k - d_k_oop * normal
        d_i_bisector = np.dot(d_i_ip, bisector)
        d_k_bisector = np.dot(d_k_ip, bisector)

        if d_i_bisector * d_k_bisector < 0:
            return "symmetric deformation"  # in-phase angle closing/opening
        else:
            return "asymmetric deformation"  # both terminals rock same direction


def _sub_classify_stretching(
    displacement: np.ndarray,
    positions: np.ndarray,
    per_bond_scores: list,
    sig_fraction: float = 0.15,
) -> str:
    """Sub-classify a stretching mode as symmetric or asymmetric.

    Compares the sign of signed bond-stretching projections for bonds with
    significant contributions.  When all contributing bonds elongate (or all
    shorten) together the mode is symmetric; when some elongate while others
    shorten it is asymmetric.

    A single dominant bond returns '' because symmetric/asymmetric only has
    meaning when at least two bonds participate.

    Args:
        displacement:    (n_atoms, 3) Cartesian displacement for one mode.
        positions:       (n_atoms, 3) equilibrium positions.
        per_bond_scores: List of ((i, j), score) from
                         ``_project_mode_onto_stretches``.
        sig_fraction:    Minimum fraction of the total stretch score for a bond
                         to be considered significant (default 0.15).

    Returns:
        'symmetric', 'asymmetric', or '' (single dominant bond).
    """
    if not per_bond_scores:
        return ""

    total = sum(s for _, s in per_bond_scores)
    if total < 1e-10:
        return ""

    significant = [(bond, s) for bond, s in per_bond_scores if s / total > sig_fraction]

    if len(significant) < 2:
        return ""  # Single bond dominates — no sym/asym distinction

    # Signed projection along each bond direction
    projs = []
    for (i, j), _ in significant:
        r_ij = positions[j] - positions[i]
        r_norm = np.linalg.norm(r_ij)
        if r_norm < 1e-10:
            continue
        e_ij = r_ij / r_norm
        delta = np.dot(displacement[j] - displacement[i], e_ij)
        projs.append(delta)

    if len(projs) < 2:
        return ""

    n_pos = sum(1 for p in projs if p > 0)
    n_neg = len(projs) - n_pos

    if n_pos == 0 or n_neg == 0:
        return "symmetric"
    return "asymmetric"


# ============================================================================
# Multi-molecule helpers
# ============================================================================

COLLECTIVE_MODE_THRESHOLD = 0.80  # fraction above which a mode is intra-molecular


def _build_atom_to_fragment(fragments: list, n_atoms: int) -> np.ndarray:
    """Return array mapping atom index → fragment index."""
    mapping = np.full(n_atoms, -1, dtype=int)
    for frag_id, atom_list in enumerate(fragments):
        for atom_idx in atom_list:
            mapping[atom_idx] = frag_id
    return mapping


def _compute_fragment_contributions(
    displacement: np.ndarray, masses: np.ndarray, fragments: list
) -> dict:
    """Partition a mode's kinetic energy across molecular fragments.

    Uses mass-weighted squared displacement: KE_frag = sum_i(m_i * |d_i|^2)
    for atoms i in that fragment, normalized so all fractions sum to 1.

    Args:
        displacement: (n_atoms, 3) Cartesian displacement for one mode.
        masses: (n_atoms,) atomic masses in amu.
        fragments: List of atom-index lists, one per fragment.

    Returns:
        Dict ``{fragment_id: fraction}`` summing to 1.0.
    """
    sq_disp = np.sum(displacement**2, axis=1)  # (n_atoms,)
    weighted = masses * sq_disp
    total = np.sum(weighted)
    if total < 1e-30:
        return {i: 1.0 / len(fragments) for i in range(len(fragments))}
    return {
        i: float(np.sum(weighted[frag_atoms]) / total) for i, frag_atoms in enumerate(fragments)
    }


def _detect_hbond_character(
    dominant_atoms: list, atom_to_frag: np.ndarray, symbols: list, positions: np.ndarray
) -> bool:
    """Check if dominant atoms from different fragments form an H-bond pattern.

    Looks for D-H...A where D/A are O or N, H is bonded to D, and
    H...A distance is 1.2–2.5 Å.
    """
    if len(dominant_atoms) < 2:
        return False
    hbond_donors_acceptors = {"O", "N", "F"}
    frags = set(atom_to_frag[a] for a in dominant_atoms)
    if len(frags) < 2:
        return False
    for a in dominant_atoms:
        if symbols[a] == "H":
            for b in dominant_atoms:
                if b == a:
                    continue
                if symbols[b] in hbond_donors_acceptors:
                    if atom_to_frag[a] != atom_to_frag[b]:
                        dist = np.linalg.norm(positions[a] - positions[b])
                        if 1.2 < dist < 2.5:
                            return True
        elif symbols[a] in hbond_donors_acceptors:
            for b in dominant_atoms:
                if b == a:
                    continue
                if symbols[b] in hbond_donors_acceptors:
                    if atom_to_frag[a] != atom_to_frag[b]:
                        dist = np.linalg.norm(positions[a] - positions[b])
                        if 2.0 < dist < 3.5:
                            return True
    return False


# ============================================================================
# Main Classification
# ============================================================================


_ACCFG_CITATION_SHOWN = False


def _log_accfg_citation():
    """Print the AccFG citation once when peak allocation to functional groups is used."""
    global _ACCFG_CITATION_SHOWN
    if _ACCFG_CITATION_SHOWN:
        return
    _ACCFG_CITATION_SHOWN = True
    log_message("")
    log_message("Vibrational peak allocation to functional groups uses the AccFG-style approach.")
    log_message("If you use it, please cite:")
    log_message(
        "  X. Liu, S. Swaminathan, D. Zubarev, et al., "
        "J. Chem. Inf. Model. 65 (2025) 8593-8602. "
        "doi:10.1021/acs.jcim.5c01317"
    )
    log_message("")


def classify_vibrational_modes(
    results: dict, mixed_threshold: float = 0.6, vib_threshold: float = 50.0
) -> dict:
    """Classify vibrational modes into stretching, bending, torsion, or mixed.

    Implements the full protocol: builds molecular topology, projects each normal
    mode onto internal coordinates (bond stretches, angle bends, dihedral torsions),
    and assigns the dominant motion type with chemical labels.

    Args:
        results: Dictionary from compute_ir_spectrum() containing:
            positions, symbols, masses, normal_modes, frequencies,
            intensities_normalized, numbers
        mixed_threshold: Fraction threshold for dominant type (default: 0.6)
        vib_threshold: Frequency threshold to skip trans/rot modes (cm-1)

    Returns:
        Dictionary with:
            modes: List of per-mode classification dicts
            topology: Molecular topology info
            n_stretching, n_bending, n_torsion, n_mixed: Counts
    """
    positions = np.asarray(results["positions"])
    symbols = results["symbols"]
    masses = np.asarray(results["masses"])
    normal_modes = np.asarray(results["normal_modes"])
    frequencies = np.asarray(results["frequencies"])
    intensities = np.asarray(results["intensities_normalized"])
    numbers = np.asarray(results["numbers"])

    n_atoms = len(symbols)

    charge = int(results.get("charge_total", 0))

    # Step 1: Build topology (or use pre-built if provided in results)
    topology = results.get("topology")
    if topology is None:
        topology = _build_topology(positions, numbers, symbols, charge=charge)

    # Step 1b: Detect molecular fragments (reuse bonds from topology)
    from .topology import _connected_components

    fragments = _connected_components(n_atoms, topology["bonds"])
    is_multi = len(fragments) > 1
    atom_to_frag = _build_atom_to_fragment(fragments, n_atoms) if is_multi else None

    if is_multi:
        log_header("MULTI-MOLECULE SYSTEM DETECTED")
        log_message(f"  Fragments: {len(fragments)}")
        for fid, frag in enumerate(fragments):
            frag_syms = [symbols[a] for a in frag]
            formula = "".join(sorted(set(frag_syms), key=lambda s: (s != "C", s != "H", s)))
            counts_str = "".join(
                f"{el}{frag_syms.count(el)}" if frag_syms.count(el) > 1 else el
                for el in sorted(set(frag_syms), key=lambda s: (s != "C", s != "H", s))
            )
            log_message(
                f"  mol {fid}: {counts_str} ({len(frag)} atoms, " f"indices {frag[0]}-{frag[-1]})"
            )

    # Step 1c: Molecule-level functional group identification (AccFG-style)
    from .functional_groups import get_functional_group_for_atoms, identify_functional_groups

    fg_map = identify_functional_groups(symbols, topology)
    if fg_map:
        _log_accfg_citation()

    # Log topology with bond orders
    bond_orders = topology.get("bond_orders", {})
    n_single = sum(1 for o in bond_orders.values() if o == 1)
    n_double = sum(1 for o in bond_orders.values() if o == 2)
    n_triple = sum(1 for o in bond_orders.values() if o == 3)
    log_info(
        f"Topology: {len(topology['bonds'])} bonds "
        f"({n_single} single, {n_double} double, {n_triple} triple), "
        f"{len(topology['angles'])} angles, "
        f"{len(topology['dihedrals'])} dihedrals"
    )

    # Step 2: Convert modes to Cartesian displacements
    cartesian_modes = _mode_to_cartesian_displacements(normal_modes, masses)

    # Classify each vibrational mode
    mode_classifications = []
    counts = {"stretching": 0, "bending": 0, "torsion": 0, "mixed": 0, "intermolecular": 0}

    for k in range(len(frequencies)):
        freq = frequencies[k]

        # Skip translation/rotation modes
        if abs(freq) < vib_threshold:
            continue

        # Get displacement for this mode, reshape to (n_atoms, 3)
        disp = cartesian_modes[:, k].reshape(n_atoms, 3)

        # Step 3: Project onto internal coordinates
        s_score, per_bond = _project_mode_onto_stretches(disp, positions, topology["bonds"])
        b_score, per_angle = _project_mode_onto_bends(disp, positions, topology["angles"])
        t_score, per_dihedral = _project_mode_onto_torsions(disp, positions, topology["dihedrals"])

        # Step 4-5: Classify
        mode_type, scores = _classify_single_mode(s_score, b_score, t_score, mixed_threshold)

        # Step 6-7: Identify atoms and label
        atoms, label = _identify_participating_atoms(
            disp, positions, topology, mode_type, symbols, per_bond, per_angle, per_dihedral
        )

        # Step 8: Sub-classify stretching and bending
        sub_type = ""
        if mode_type == "stretching" and per_bond:
            stretch_sub = _sub_classify_stretching(disp, positions, per_bond)
            if stretch_sub:
                sub_type = stretch_sub
                label = label.replace("stretch", f"{stretch_sub} stretch")
        elif mode_type == "bending" and per_angle:
            best_angle = max(per_angle, key=lambda x: x[1])
            sub_type = _sub_classify_bending(disp, positions, best_angle[0])
            if sub_type:
                label = label.replace("bend", sub_type)

        counts[mode_type] += 1

        # ── Multi-molecule: fragment attribution ──────────────────────
        fragment_id = None
        fragment_contribs = None
        is_intermolecular = False

        if is_multi:
            fragment_contribs = _compute_fragment_contributions(disp, masses, fragments)
            dominant_frag = max(fragment_contribs, key=fragment_contribs.get)
            if fragment_contribs[dominant_frag] >= COLLECTIVE_MODE_THRESHOLD:
                fragment_id = dominant_frag
            else:
                fragment_id = "collective"
                is_intermolecular = True
                counts["intermolecular"] += 1

            # Append fragment tag to label
            if is_intermolecular:
                atom_ints = [int(a) for a in atoms]
                if _detect_hbond_character(atom_ints, atom_to_frag, symbols, positions):
                    label += " [intermolecular H-bond]"
                else:
                    label += " [intermolecular]"
            else:
                label += f" [mol {fragment_id}]"

        # Try AccFG-style molecule-level FG first, fall back to per-mode
        functional_group = None
        if fg_map:
            functional_group = get_functional_group_for_atoms(
                fg_map, [int(a) for a in atoms], symbols, topology
            )
        if not functional_group:
            functional_group = _get_functional_group_context(
                [(int(a), symbols[a]) for a in atoms], symbols, topology, mode_type
            )

        mode_classifications.append(
            {
                "mode_index": int(k),
                "frequency": float(freq),
                "intensity": float(intensities[k]),
                "type": mode_type,
                "scores": scores,
                "dominant_atoms": [(int(a), symbols[a]) for a in atoms],
                "label": label,
                "sub_type": sub_type,
                "functional_group": functional_group,
                "fragment_id": fragment_id,
                "fragment_contributions": fragment_contribs,
                "is_intermolecular": is_intermolecular,
            }
        )

    # Log summary
    log_header("MODE CLASSIFICATION SUMMARY")
    log_message(f"  Stretching: {counts['stretching']}")
    log_message(f"  Bending:    {counts['bending']}")
    log_message(f"  Torsion:    {counts['torsion']}")
    log_message(f"  Mixed:      {counts['mixed']}")
    if is_multi:
        log_message(f"  Intermolecular: {counts['intermolecular']}")

    return {
        "modes": mode_classifications,
        "topology": topology,
        "n_stretching": counts["stretching"],
        "n_bending": counts["bending"],
        "n_torsion": counts["torsion"],
        "n_mixed": counts["mixed"],
        "is_multimolecular": is_multi,
        "n_fragments": len(fragments),
        "fragments": fragments if is_multi else None,
        "n_intermolecular": counts["intermolecular"],
    }


# ============================================================================
# Main IR Calculation
# ============================================================================


def load_multiframe_xyz(xyz_file: str) -> list:
    """Load all conformers from a multi-frame XYZ file.

    Args:
        xyz_file: Path to XYZ file (single or multi-frame)

    Returns:
        List of structure dictionaries
    """
    try:
        # Try ASE first (handles multi-frame XYZ)
        from ase import io

        atoms_list = io.read(xyz_file, index=":")

        # Convert to list if single structure
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        structures = []
        for atoms in atoms_list:
            from .utils import ase_atoms_to_structure

            structures.append(ase_atoms_to_structure(atoms))

        return structures
    except ImportError:
        # Fallback: try simple parser
        structures = []
        try:
            with open(xyz_file, "r") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue

                n_atoms = int(lines[i].strip())
                comment = lines[i + 1].strip()

                symbols = []
                positions = []
                for j in range(n_atoms):
                    parts = lines[i + 2 + j].split()
                    symbols.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

                from .utils import create_structure

                structures.append(create_structure(positions=jnp.array(positions), symbols=symbols))

                i += n_atoms + 2

            return structures
        except Exception as e:
            log_info(f"Error parsing multi-frame XYZ: {e}")
            # Try single structure load as fallback
            return [load_structure(xyz_file)]


def compute_ir_spectrum(
    xyz_file: str,
    potential_name: str = "so3lr",
    model_path: Optional[str] = None,
    temperature: float = 300.0,
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    use_float64: bool = False,
    conformer_index: Optional[int] = None,
    optimize: bool = True,
    fmax: float = 0.01,
    classify: bool = True,
    fd_hessian: bool = False,
    fd_displacement: float = 0.01,
    potential_options: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
) -> Dict:
    """Compute IR spectrum from XYZ file using an ML potential.

    This function:
    1. Loads the structure from XYZ file
    2. Initializes the chosen potential
    3. Computes analytical Hessian
    4. Extracts partial charges (if supported by the potential)
    5. Computes normal modes
    6. Calculates IR intensities using dipole derivatives

    If the potential does not support partial charges, frequencies and
    normal modes are still computed but IR intensities are set to zero
    and a warning is issued.

    Args:
        xyz_file: Path to input XYZ file (single or multi-frame)
        potential_name: Registered potential name (default: ``"so3lr"``).
        model_path: Optional path to custom model (default: use pretrained)
        temperature: Temperature in Kelvin for thermal populations
        charge: Total molecular charge
        lr_cutoff: Long-range cutoff in Angstrom (1000 for gas phase)
        use_float64: If True, use float64 precision; otherwise use float32
            (float64 is strongly recommended for Hessian accuracy).
        conformer_index: For multi-frame XYZ, which conformer to use
            (0-indexed). ``None`` selects the first frame.
        optimize: If True, optimize the geometry before computing the
            Hessian (recommended).
        fmax: Force-convergence threshold for the pre-Hessian
            optimization, in eV/Å (ignored when ``optimize=False``).
        classify: If True, classify each vibrational mode and assign it
            to a functional group via AccFG-style graph matching.
        fd_hessian: If True, approximate the Hessian by finite
            differences instead of JAX autodiff (use when autodiff is
            numerically unstable for the chosen potential).
        fd_displacement: Cartesian displacement step in Å for the
            finite-differences Hessian (default 0.01).
        potential_options: Optional dict of potential-specific kwargs
            forwarded to the wrapper constructor (e.g.
            ``{"mace_foundation": "off", "mace_model": "small"}``
            for MACE, or ``{"dxtb_method": "gfn2"}`` for dxtb).
        atom_indices: Optional list of 0-based atom indices selecting a
            subset of the molecule for a partial Hessian (e.g. solute
            atoms in an explicit solvent). When given, the full system
            is still used for the energy/charges and the geometry
            optimization, but only the selected atoms are displaced for
            the Hessian — yielding a (3K, 3K) sub-block. The returned
            ``positions``/``symbols``/``masses``/``numbers``/``charges``
            keys refer to the selected subset; the full structure is
            kept under ``full_positions``/``full_symbols``/
            ``full_numbers``/``full_masses`` and the selection itself
            under ``atom_indices``.

    Returns:
        Dictionary with at least:

        - ``frequencies`` — ``(n_modes,)`` wavenumbers in cm⁻¹.
        - ``intensities`` — ``(n_modes,)`` IR intensities.
        - ``normal_modes`` — ``(3*n_atoms, n_modes)`` mode vectors.
        - ``thermal_populations`` — Boltzmann populations at ``temperature``.
        - ``hessian`` — ``(3*n_atoms, 3*n_atoms)`` mass-weighted Hessian (eV/Å²/amu).
        - ``charges`` — ``(n_atoms,)`` partial charges (``None`` if unavailable).
        - ``positions`` — ``(n_atoms, 3)`` (optimized) geometry in Å.
        - ``energy`` — total potential energy in eV.

    Example:
        >>> from mars.ir import compute_ir_spectrum, save_ir_plot
        >>> result = compute_ir_spectrum(
        ...     "molecule.xyz", potential_name="so3lr",
        ...     fmax=0.005, broadening=10.0,
        ... )  # doctest: +SKIP
        >>> save_ir_plot(result, "ir.png", freq_range=(400, 4000))
    """
    # Load structure(s)
    log_header("IR SPECTRUM CALCULATION")
    log_info(f"Loading structure from {xyz_file}...")

    structures = load_multiframe_xyz(xyz_file)
    structure, conformer_index = _select_conformer(structures, conformer_index)
    positions = structure["positions"]
    numbers = structure["numbers"]
    symbols = structure["symbols"]
    masses = get_atomic_masses(numbers)
    n_atoms_full = len(symbols)

    atom_indices = _normalize_atom_indices(atom_indices, n_atoms_full)
    if atom_indices is not None:
        log_info(
            f"Partial IR Hessian: {len(atom_indices)} of {n_atoms_full} atoms selected "
            f"(indices {atom_indices[0]}–{atom_indices[-1]})"
        )

    dtype = _resolve_dtype(use_float64)

    # Initialize potential
    log_info(f"Initializing {potential_name} potential...")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        potential = _build_ir_potential(
            potential_name,
            numbers,
            lr_cutoff,
            charge,
            model_path,
            dtype,
            compute_charges=True,
            potential_options=potential_options,
        )
        potential.initialize(positions)
        result = potential.build_energy_fn(with_charges=True)

    if isinstance(result, tuple):
        energy_fn, charges_fn = result
    else:
        energy_fn, charges_fn = result, None

    has_charges = charges_fn is not None
    if not has_charges:
        warnings.warn(
            f"The '{potential_name}' potential does not support partial charges. "
            "IR intensities cannot be computed — only frequencies and normal "
            "modes will be reported.",
            stacklevel=2,
        )
        log_info(
            f"Warning: {potential_name} potential does not support partial "
            "charges. IR intensities cannot be computed — only frequencies "
            "and normal modes will be reported."
        )

    # JIT-compile energy function for speed (always on)
    log_info("JIT-compiling energy function...")
    energy_fn_jit = jax.jit(energy_fn)
    charges_fn_jit = jax.jit(charges_fn) if has_charges else None

    # Optimize geometry if requested
    if optimize:
        log_message("")
        log_info("Optimizing geometry before Hessian calculation...")
        log_info(f"Force convergence criterion: {fmax:.3g} eV/Å")

        from .optimizer import optimize_single

        initial_energy = float(energy_fn_jit(positions))
        log_info(f"Initial energy: {initial_energy * EV_TO_KCALMOL:.2f} kcal/mol")

        positions_opt, energy_opt, opt_info = optimize_single(
            positions, energy_fn_jit, fmax=fmax, maxiter=1000
        )

        # Check convergence
        max_force = opt_info["max_force"]
        converged = opt_info["converged"]
        n_iter = opt_info["iterations"]

        log_info(
            f"Optimization {'converged' if converged else 'did not converge'} in {n_iter} steps"
        )
        log_info(f"Final max force: {max_force:.3g} eV/Å")
        log_info(f"Optimized energy: {energy_opt * EV_TO_KCALMOL:.2f} kcal/mol")
        log_info(f"Energy change: {(energy_opt - initial_energy) * EV_TO_KCALMOL:.2f} kcal/mol")

        if not converged:
            log_info(
                "Warning: Optimization did not converge. Results may not be at a true minimum."
            )
            log_info("         This may lead to imaginary frequencies in the IR spectrum.")

        # Use optimized positions
        positions = positions_opt
        energy = energy_opt
    else:
        # Compute energy
        log_message("")
        log_info("Computing energy...")
        energy = float(energy_fn_jit(positions))
        log_info(f"Energy: {energy * EV_TO_KCALMOL:.2f} kcal/mol")

    # When a partial Hessian is requested, slice the per-atom quantities
    # used downstream of the Hessian (mass-weighting, normal modes,
    # intensities, classification, save functions) to the selected
    # atoms. The energy/charges functions and the geometry optimization
    # always see the full system.
    use_partial = atom_indices is not None
    if use_partial:
        idx_arr = jnp.asarray(atom_indices, dtype=jnp.int32)
        sel_positions = positions[idx_arr]
        sel_numbers = np.asarray(numbers)[atom_indices]
        sel_symbols = [symbols[i] for i in atom_indices]
        sel_masses = (
            masses[idx_arr]
            if hasattr(masses, "shape")
            else jnp.array([masses[i] for i in atom_indices])
        )
    else:
        sel_positions = positions
        sel_numbers = numbers
        sel_symbols = symbols
        sel_masses = masses

    # Compute Hessian — fall back to FD for potentials that use pure_callback
    # (e.g. dxtb, MACE-torch) because JAX forward-mode (JVP) cannot propagate
    # through pure_callback, making jax.hessian impossible.
    use_fd = fd_hessian or not potential.analytical_hessian
    log_message("")
    partial_tag = f" [partial: {len(atom_indices)}/{n_atoms_full} atoms]" if use_partial else ""

    def _compute_fd_hessian():
        """Run the finite-differences Hessian (and dipole derivatives) pass."""
        fd_charges_fn = charges_fn_jit if has_charges else None
        log_info(
            "Computing finite-differences Hessian"
            + (" and dipole derivatives" if fd_charges_fn is not None else "")
            + f" in a single pass (displacement={fd_displacement} Å){partial_tag}..."
        )
        return compute_hessian_and_dipole_derivs_fd(
            energy_fn_jit,
            fd_charges_fn,
            positions,
            displacement=fd_displacement,
            atom_indices=atom_indices,
        )

    if use_fd:
        if not fd_hessian:
            log_info(
                f"Note: '{potential_name}' uses a PyTorch callback and does not "
                "support analytical Hessians. Switching to finite differences automatically."
            )
        hessian, fd_dipole_derivs = _compute_fd_hessian()
    else:
        an_charges_fn = charges_fn_jit if has_charges else None
        msg = (
            "Computing analytical Hessian and dipole derivatives in a single pass"
            if an_charges_fn is not None
            else "Computing analytical Hessian"
        )
        log_info(msg + f" (JIT-compiled){partial_tag}...")
        try:
            hessian, fd_dipole_derivs = compute_hessian_and_dipole_derivs(
                energy_fn_jit,
                an_charges_fn,
                positions,
                use_jit=True,
                atom_indices=atom_indices,
            )
        except Exception as exc:
            # The analytical Hessian materialises a (3N, 3N) dense forward-mode
            # pass and can exhaust GPU/host memory for large systems. On OOM,
            # warn and fall back to the memory-light finite-differences path.
            if not _is_jax_oom(exc):
                raise
            log_warning(
                "Out of memory while computing the analytical Hessian; "
                "falling back to a finite-differences Hessian (--fd-hessian). "
                "Use --fd-hessian explicitly to skip this attempt next time."
            )
            jax.clear_caches()
            hessian, fd_dipole_derivs = _compute_fd_hessian()
    log_info(f"Hessian shape: {hessian.shape}")

    # Mass-weight Hessian
    log_info("Mass-weighting Hessian...")
    hessian_mw = mass_weight_hessian(hessian, sel_masses)

    # Compute normal modes
    log_info("Computing normal modes...")
    frequencies, normal_modes, thermal_pops = compute_normal_modes(
        hessian_mw, sel_masses, temperature
    )

    # Identify vibrational modes (exclude translations/rotations)
    # Translation/rotation modes have frequencies near zero
    vib_threshold = 50.0  # cm⁻¹
    vib_mask = jnp.abs(frequencies) > vib_threshold
    n_vib = jnp.sum(vib_mask)

    log_info(f"Found {n_vib} vibrational modes (threshold: {vib_threshold} cm⁻¹)")
    if n_vib > 0:
        log_info(
            f"Frequency range: {jnp.min(frequencies[vib_mask]):.0f} to "
            f"{jnp.max(frequencies[vib_mask]):.0f} cm⁻¹"
        )

    # Compute IR intensities (requires charges)
    if has_charges:
        # Compute initial dipole moment: μ = Σ q_i · r_i over the full system
        initial_charges = charges_fn_jit(positions)
        initial_dipole = jnp.sum(initial_charges[:, None] * positions, axis=0)
        log_info(
            f"Initial dipole moment: ({float(initial_dipole[0]):.4f}, {float(initial_dipole[1]):.4f}, {float(initial_dipole[2]):.4f}) e·Å"
        )
        log_info(f"Initial dipole magnitude: {float(jnp.linalg.norm(initial_dipole)):.4f} e·Å")

        log_message("")
        log_info("Computing dipole derivatives...")
        if fd_dipole_derivs is not None:
            # Already computed alongside the FD Hessian — reuse
            dipole_derivs = fd_dipole_derivs
        else:
            dipole_derivs = compute_dipole_derivatives(
                charges_fn_jit, positions, atom_indices=atom_indices
            )

        log_info("Computing IR intensities...")
        intensities = compute_ir_intensities(normal_modes, dipole_derivs, sel_masses)

        max_intensity = jnp.max(intensities[vib_mask]) if jnp.any(vib_mask) else 1.0
        intensities_norm = intensities / (max_intensity + 1e-10)
    else:
        n_modes = 3 * len(sel_symbols)
        intensities = np.zeros(n_modes)
        intensities_norm = np.zeros(n_modes)

    # Build results
    results = {
        "frequencies": np.array(frequencies),
        "intensities": np.array(intensities),
        "intensities_normalized": np.array(intensities_norm),
        "normal_modes": np.array(normal_modes),
        "thermal_populations": np.array(thermal_pops),
        "hessian": np.array(hessian),
        "charges": (
            np.array(charges_fn_jit(positions))[atom_indices]
            if (has_charges and use_partial)
            else (np.array(charges_fn_jit(positions)) if has_charges else None)
        ),
        "positions": np.array(sel_positions),
        "energy": energy,
        "symbols": sel_symbols,
        "masses": np.array(sel_masses),
        "numbers": np.array(sel_numbers),
        "temperature": temperature,
        "charge_total": int(charge),
    }
    if use_partial:
        results.update(
            {
                "atom_indices": list(atom_indices),
                "full_positions": np.array(positions),
                "full_symbols": list(symbols),
                "full_numbers": np.array(numbers),
                "full_masses": np.array(masses),
                "full_charges": (np.array(charges_fn_jit(positions)) if has_charges else None),
            }
        )

    # Classify vibrational modes
    if classify:
        log_message("")
        log_info("Classifying vibrational modes...")
        classification = classify_vibrational_modes(results)
        results["mode_classification"] = classification

    # Print summary of strongest modes (with classification if available)
    log_message("")
    log_header("IR SPECTRUM SUMMARY")

    classification = results.get("mode_classification")
    mode_lookup = {}
    if classification:
        for m in classification["modes"]:
            mode_lookup[m["mode_index"]] = m

    if mode_lookup:
        log_message(
            f"{'Mode':<6} {'Freq (cm⁻¹)':<14} {'Intensity':<12} {'Type':<12} "
            f"{'Label':<30} {'Functional Group'}"
        )
        log_message("-" * 110)
    else:
        log_message(f"{'Mode':<6} {'Freq (cm⁻¹)':<14} {'Intensity':<12} {'Population':<10}")
        log_message("-" * 60)

    sorted_indices = jnp.argsort(frequencies)
    for idx in sorted_indices:
        freq = frequencies[idx]
        intensity = intensities_norm[idx]
        pop = thermal_pops[idx]
        if freq <= vib_threshold:
            continue
        m_info = mode_lookup.get(int(idx))
        if not has_charges:
            if m_info:
                fg = m_info.get("functional_group", "")
                log_message(
                    f"{idx:<6} {freq:>12.1f}   {'--':>10}   "
                    f"{m_info['type']:<12} {m_info['label']:<30} {fg}"
                )
            else:
                log_message(f"{idx:<6} {freq:>12.1f}   {'--':>10}   {pop:>8.2f}")
        elif intensity > 0.01:
            if m_info:
                fg = m_info.get("functional_group", "")
                log_message(
                    f"{idx:<6} {freq:>12.1f}   {intensity:>10.3f}   "
                    f"{m_info['type']:<12} {m_info['label']:<30} {fg}"
                )
            else:
                log_message(f"{idx:<6} {freq:>12.1f}   {intensity:>10.3f}   {pop:>8.2f}")

    log_message("=" * 110 if mode_lookup else "=" * 60)

    return results


def compute_ir_from_md(
    xyz_file: str,
    potential_name: str = "so3lr",
    model_path: Optional[str] = None,
    temperature: float = 300.0,
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    use_float64: bool = False,
    conformer_index: Optional[int] = None,
    nvt_time_ps: float = 5.0,
    nve_time_ps: float = 25.0,
    dt_fs: float = 0.5,
    dipole_save_fs: float = 2.5,
    trajectory_file: Optional[str] = None,
    chop: Optional[int] = 10,
    window: str = "hann",
    random_seed: int = 0,
    potential_options: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
) -> Dict:
    """Compute IR spectrum from MD trajectory using dipole autocorrelation.

    This function uses jax-md to run:
    1. NVT equilibration (default 5 ps) with charges
    2. NVE production (default 25 ps) with charges
    3. Computes IR via Wiener-Kinchin theorem and dipole autocorrelation

    Partial charges are **required** for MD-based IR.  If the chosen
    potential does not support charges, a ``RuntimeError`` is raised.

    Args:
        xyz_file: Path to an input XYZ file (single- or multi-frame).
        potential_name: Registered potential name; must provide partial
            charges (default ``"so3lr"``).
        model_path: Optional path to custom model weights.
        temperature: Temperature in K (default 300).
        charge: Total molecular charge.
        lr_cutoff: Long-range cutoff in Å (default 1000 for gas phase).
        use_float64: If True, use float64 precision (forced on internally
            anyway — included for API parity).
        conformer_index: For multi-frame XYZ, which conformer to use
            (0-indexed).
        nvt_time_ps: NVT equilibration time in ps (default 5.0).
        nve_time_ps: NVE production time in ps (default 25.0).
        dt_fs: MD timestep in fs (default 0.5).
        dipole_save_fs: Save the dipole moment every N fs (default 2.5).
        trajectory_file: Optional path to save the NVE trajectory in
            HDF5 format.
        chop: Number of segments to split the trajectory into for
            averaging — improves spectral statistics (default 10).
        window: Window function for the FFT autocorrelation
            (``"hann"``, ``"hamming"``, ``"blackman"``, ``"bartlett"``,
            or ``None``).
        random_seed: Seed for MD initialization (default 0).
        potential_options: Optional dict of potential-specific kwargs
            (see :func:`compute_ir_spectrum`).
        atom_indices: Optional 0-based atom subset used to track the
            dipole. The full system still propagates dynamics; the
            dipole μ(t) = Σ_{i ∈ atom_indices} q_i(t) r_i(t) is summed
            only over the selected atoms. Useful for isolating the IR
            signal of a solute in explicit solvent.

    Returns:
        Dictionary with at least:

        - ``frequencies`` — ``(n_points,)`` wavenumbers in cm⁻¹.
        - ``intensities`` — ``(n_points,)`` IR intensities from the dipole autocorrelation.
        - ``intensities_error`` — standard error across segments (when ``chop > 1``).
        - ``trajectory`` — MD trajectory data.
        - ``dipole_moments`` — dipole-moment time series.

    Raises:
        RuntimeError: If the chosen potential does not provide partial
            charges (MD-based IR requires them).

    Example:
        >>> from mars.ir import compute_ir_from_md
        >>> result = compute_ir_from_md(
        ...     "molecule.xyz",
        ...     potential_name="so3lr",
        ...     nvt_time_ps=5.0,
        ...     nve_time_ps=25.0,
        ...     chop=10,
        ...     window="hann",
        ... )  # doctest: +SKIP
    """
    import jax.random as random
    from jax_md import simulate, space, units

    unit = units.metal_unit_system()
    # Load structure
    log_header("MD-BASED IR SPECTRUM CALCULATION")
    log_info(f"Loading structure from {xyz_file}...")

    structures = load_multiframe_xyz(xyz_file)
    structure, conformer_index = _select_conformer(structures, conformer_index)
    positions = structure["positions"]
    numbers = structure["numbers"]
    symbols = structure["symbols"]
    masses = get_atomic_masses(numbers)
    n_atoms = len(symbols)

    dtype = _resolve_dtype(use_float64, force_float64_for_md=True)
    positions = jnp.array(positions, dtype=dtype)
    masses = jnp.array(masses, dtype=dtype)

    atom_indices = _normalize_atom_indices(atom_indices, n_atoms)
    if atom_indices is not None:
        log_info(
            f"Tracking dipole over {len(atom_indices)} of {n_atoms} atoms "
            f"(indices {atom_indices[0]}–{atom_indices[-1]})"
        )
        dipole_atoms_idx = jnp.asarray(atom_indices, dtype=jnp.int32)
    else:
        dipole_atoms_idx = None

    # Initialize potential with charges
    log_info(f"Initializing {potential_name} potential with partial charges...")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        potential = _build_ir_potential(
            potential_name,
            numbers,
            lr_cutoff,
            charge,
            model_path,
            dtype,
            compute_charges=True,
            potential_options=potential_options,
        )
        potential.initialize(positions)
        result = potential.build_energy_fn(with_charges=True)

    if isinstance(result, tuple):
        energy_fn, charges_fn = result
    else:
        energy_fn, charges_fn = result, None

    if charges_fn is None:
        raise RuntimeError(
            f"MD-based IR requires partial charges but the '{potential_name}' "
            "potential does not support them. Use a potential that provides "
            "partial charges (e.g. so3lr), or use Hessian-based IR instead "
            "(remove --md flag)."
        )

    # Allocate neighbor lists
    log_info("Allocating neighbor lists...")
    nbr_state = potential.allocate_neighbors(positions)

    # Compute initial energy with neighbor lists
    nbr_kwargs = potential.get_neighbor_kwargs(nbr_state)
    initial_energy = float(energy_fn(positions, **nbr_kwargs))
    log_info(f"Initial energy: {initial_energy * EV_TO_KCALMOL:.2f} kcal/mol")

    # Convert units
    dt_ps = dt_fs / 1000.0  # fs to ps
    dt = dt_ps * unit["time"]
    kT = temperature * unit["temperature"]

    log_message("")
    log_header("NVT EQUILIBRATION")
    log_info(f"Temperature: {temperature:.0f} K")
    log_info(f"Time: {nvt_time_ps:.3g} ps")
    log_info(f"Timestep: {dt_fs:.3g} fs")

    # NVT simulation using Langevin dynamics
    nvt_steps = int(nvt_time_ps / dt_ps)
    gamma = 0.01  # friction coefficient in 1/ps

    # Progress tracking: run in chunks for progress bar
    nvt_save_interval = max(100, nvt_steps // 100)  # Update progress ~100 times
    n_nvt_frames = nvt_steps // nvt_save_interval
    nvt_steps = n_nvt_frames * nvt_save_interval  # Ensure divisible

    log_info(
        f"Running {nvt_steps} NVT steps in {n_nvt_frames} chunks of {nvt_save_interval} steps..."
    )
    _, shift_fn = space.free()
    # Initialize Langevin simulator
    init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma)

    # Initialize state
    key = random.PRNGKey(random_seed)
    state = init_fn(
        key, positions, mass=masses, kT=kT * 2
    )  # Start with higher temp for faster equilibration

    # JIT-compiled inner NVT loop
    @jax.jit
    def run_nvt_inner(jax_state, nbrs):
        def body(i, carry):
            s, nbrs = carry
            # Update neighbor lists
            nbrs = potential.update_neighbors(s.position, nbrs)
            nbr_kwargs = potential.get_neighbor_kwargs(nbrs)
            # Apply NVT step with neighbor kwargs
            s = apply_fn(s, **nbr_kwargs)
            return (s, nbrs)

        return jax.lax.fori_loop(0, nvt_save_interval, body, (jax_state, nbrs))

    # Run NVT equilibration with progress bar
    for frame in tqdm(range(n_nvt_frames), desc="NVT Equilibration", unit="chunk"):
        state, nbr_state = run_nvt_inner(state, nbr_state)

        # Check for neighbor list overflow and reallocate if needed
        overflow = False
        if isinstance(nbr_state, tuple):
            # SO3LR: tuple of (nbrs, nbrs_lr)
            overflow = nbr_state[0].did_buffer_overflow or nbr_state[1].did_buffer_overflow
        else:
            # Single neighbor list
            overflow = nbr_state.did_buffer_overflow

        if overflow:
            tqdm.write(f"  Neighbor list overflow at chunk {frame}, reallocating...")
            nbr_state = potential.allocate_neighbors(state.position)

    # Get equilibrated positions and check temperature
    positions_eq = state.position
    nbr_kwargs = potential.get_neighbor_kwargs(nbr_state)
    eq_energy = float(energy_fn(positions_eq, **nbr_kwargs))

    # Calculate final temperature from kinetic energy
    # KE = 0.5 * sum(m_i * v_i^2), T = 2*KE / (3*N*k_B)
    velocities = state.velocity
    kinetic_energy = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    final_temperature = (2.0 * kinetic_energy) / (3.0 * n_atoms * KB_EV_PER_K)

    log_info(f"Equilibration complete. Final energy: {eq_energy * EV_TO_KCALMOL:.2f} kcal/mol")
    log_info(f"Temperature: {float(final_temperature):.0f} K (target: {temperature:.0f} K)")
    log_info(f"Temperature deviation: {abs(float(final_temperature) - temperature):.1f} K")

    log_message("")
    log_header("NVE PRODUCTION RUN")
    log_info(f"Time: {nve_time_ps:.3g} ps")
    log_info(f"Timestep: {dt_fs:.3g} fs")

    # NVE simulation using velocity Verlet
    nve_steps = int(nve_time_ps / dt_ps)

    # Calculate save interval: save dipole every dipole_save_fs femtoseconds
    save_every = max(1, int(dipole_save_fs / dt_fs))

    # Ensure nve_steps is divisible by save_every
    n_nve_frames = nve_steps // save_every
    nve_steps = n_nve_frames * save_every

    log_info(
        f"Running {nve_steps} NVE steps, saving every {save_every} steps ({dipole_save_fs:.1f} fs)"
    )
    log_info(f"Will collect {n_nve_frames} frames for IR spectrum")

    # Initialize NVE simulator
    _, apply_fn_nve = simulate.nve(energy_fn, shift_fn, dt)

    # Initialize NVE state from equilibrated positions with velocities from NVT
    state_nve = apply_fn_nve(state)

    # Preallocate arrays for trajectory and dipoles
    total_frames = n_nve_frames
    dipole_moments = jnp.zeros((total_frames, 3), dtype=dtype)
    trajectory_positions = jnp.zeros((total_frames, n_atoms, 3), dtype=dtype)

    # JIT-compiled inner NVE loop with dipole calculation
    @jax.jit
    def run_nve_inner(jax_state, nbrs, dipoles, trajs, frame_idx):
        def body(i, carry):
            s, nbrs, dip, traj = carry
            # Update neighbor lists
            nbrs = potential.update_neighbors(s.position, nbrs)
            nbr_kwargs = potential.get_neighbor_kwargs(nbrs)
            # Apply NVE step with neighbor kwargs
            s = apply_fn_nve(s, **nbr_kwargs)

            # On last step of inner loop, save position and compute dipole
            def save_frame(state_data):
                s_local, nbrs_local, dip_local, traj_local = state_data
                nbr_kwargs_local = potential.get_neighbor_kwargs(nbrs_local)
                charges = charges_fn(s_local.position, **nbr_kwargs_local)
                if dipole_atoms_idx is not None:
                    pos_sel = s_local.position[dipole_atoms_idx]
                    q_sel = charges[dipole_atoms_idx]
                    dipole = jnp.sum(q_sel[:, None] * pos_sel, axis=0)
                else:
                    dipole = jnp.sum(charges[:, None] * s_local.position, axis=0)
                dip_local = dip_local.at[frame_idx].set(dipole)
                traj_local = traj_local.at[frame_idx].set(s_local.position)
                return dip_local, traj_local

            def no_save(state_data):
                _, _, dip_local, traj_local = state_data
                return dip_local, traj_local

            # Save only on last iteration
            dip, traj = jax.lax.cond(
                (i == save_every - 1), save_frame, no_save, (s, nbrs, dip, traj)
            )

            return (s, nbrs, dip, traj)

        return jax.lax.fori_loop(0, save_every, body, (jax_state, nbrs, dipoles, trajs))

    # Run NVE production with progress bar (frames 1..n_nve_frames, since 0 is the initial)
    for frame in tqdm(range(n_nve_frames), desc="NVE Production Run", unit="frame"):
        state_nve, nbr_state, dipole_moments, trajectory_positions = run_nve_inner(
            state_nve, nbr_state, dipole_moments, trajectory_positions, frame
        )

        # Check for neighbor list overflow and reallocate if needed
        overflow = False
        if isinstance(nbr_state, tuple):
            # SO3LR: tuple of (nbrs, nbrs_lr)
            overflow = nbr_state[0].did_buffer_overflow or nbr_state[1].did_buffer_overflow
        else:
            # Single neighbor list
            overflow = nbr_state.did_buffer_overflow

        if overflow:
            tqdm.write(f"  Neighbor list overflow at frame {frame}, reallocating...")
            nbr_state = potential.allocate_neighbors(state_nve.position)

    # Calculate final temperature and energy after NVE
    nbr_kwargs_final = potential.get_neighbor_kwargs(nbr_state)
    final_nve_energy = float(energy_fn(state_nve.position, **nbr_kwargs_final))
    velocities_nve = state_nve.velocity
    kinetic_energy_nve = 0.5 * jnp.sum(masses[:, None] * velocities_nve**2)
    final_nve_temperature = (2.0 * kinetic_energy_nve) / (3.0 * n_atoms * KB_EV_PER_K)

    log_info(f"Collected {len(dipole_moments)} trajectory frames")
    log_info(
        f"Final NVE energy: {final_nve_energy * EV_TO_KCALMOL:.2f} kcal/mol (initial: {eq_energy * EV_TO_KCALMOL:.2f} kcal/mol)"
    )
    log_info(f"NVE energy drift: {abs(final_nve_energy - eq_energy) * EV_TO_KCALMOL:.2f} kcal/mol")
    log_info(
        f"Final NVE temperature: {float(final_nve_temperature):.0f} K (target: {temperature:.0f} K)"
    )

    log_message("")
    log_header("COMPUTING IR SPECTRUM FROM DIPOLE AUTOCORRELATION")

    frequencies_cm, intensities = ir_from_dipoles(
        dipoles=dipole_moments,
        dt_fs=dipole_save_fs,
        window=window,
    )

    log_info(f"Frequency range: 0 to {frequencies_cm[-1]:.0f} cm⁻¹")
    log_info(f"Frequency resolution: {frequencies_cm[1] - frequencies_cm[0]:.1f} cm⁻¹")

    # Save NVE trajectory if requested
    if trajectory_file:
        log_message("")
        log_info(f"Saving NVE trajectory to {trajectory_file}...")
        import h5py

        with h5py.File(trajectory_file, "w") as f:
            # Save trajectory
            f.create_dataset("trajectory", data=trajectory_positions)
            f.create_dataset("dipole_moments", data=dipole_moments)

            # Metadata
            f.attrs["n_frames"] = len(trajectory_positions)
            f.attrs["n_atoms"] = n_atoms
            f.attrs["temperature"] = temperature
            f.attrs["nvt_time_ps"] = nvt_time_ps
            f.attrs["nve_time_ps"] = nve_time_ps
            f.attrs["dt_fs"] = dt_fs
            f.attrs["save_interval"] = save_every

            # Save atomic symbols
            f.create_dataset("symbols", data=np.array(symbols, dtype="S2"))

        log_info(f"NVE trajectory saved: {len(trajectory_positions)} frames")

    log_message("")
    log_header("MD-BASED IR SPECTRUM COMPLETE")

    results = {
        "frequencies": frequencies_cm,
        "intensities": intensities,
        "intensities_normalized": intensities,
        "trajectory": trajectory_positions,
        "dipole_moments": dipole_moments,
        "temperature": temperature,
        "energy": eq_energy,
        "symbols": symbols,
        "positions": np.array(positions_eq),
        "nvt_time_ps": nvt_time_ps,
        "nve_time_ps": nve_time_ps,
        "dt_fs": dt_fs,
        "save_interval": save_every,
    }

    return results


# ============================================================================
# Dipole autocorrelation tools
# ============================================================================


def ir_from_dipoles(
    dipoles,  # (T, 3)
    dt_fs: float,  # time step [fs]
    window: str = "Hann",  # window function: 'Hann', 'Hamming', 'BH', 'Gaussian', 'none'
    normalize: bool = True,
):
    """
    Compute IR spectrum from a dipole trajectory using the DACF method.

    Args:
        dipoles: Dipole trajectory in e*Angstrom, shape ``(N, 3)``.
        dt_fs: Sampling interval in femtoseconds.
        window: Window function name: 'Hann', 'Hamming', 'BH' (Blackman-Harris),
            'Gaussian', or 'none'. Default: 'Hann'.
        normalize: If True, scale the spectrum so the strongest peak is 1.

    Returns:
        frequencies: Frequencies in cm^-1 (positive only), shape ``(n_freq,)``.
        intensities: Normalised IR absorption intensities (peak = 1).
    """
    dipoles = jnp.asarray(dipoles)

    Nt = dipoles.shape[0]
    dim = dipoles.shape[1]
    Nzero = 2 ** int(jnp.ceil(jnp.log(Nt) / jnp.log(2))) - Nt

    N = 2 * (Nt + Nzero)  # Total length after zero-padding (for real FFT)

    wf = jnp.fft.fftfreq(N, dt_fs)[: N // 2]  # Frequencies in 1/fs

    # Window functions (periodic / sym=False variants)
    n = jnp.arange(Nt)
    if window.lower() != "none":
        if window.lower() == "gaussian":
            sigma = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))
            std = 4000.0 / sigma
            win = jnp.exp(-0.5 * ((n - (Nt - 1) / 2.0) / std) ** 2)
        elif window.lower() == "bh":
            # 4-term Blackman-Harris (periodic)
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            win = (
                a0
                - a1 * jnp.cos(2 * jnp.pi * n / Nt)
                + a2 * jnp.cos(4 * jnp.pi * n / Nt)
                - a3 * jnp.cos(6 * jnp.pi * n / Nt)
            )
        elif window.lower() == "hamming":
            win = 0.54 - 0.46 * jnp.cos(2 * jnp.pi * n / Nt)
        else:  # default: Hann
            win = 0.5 * (1.0 - jnp.cos(2 * jnp.pi * n / Nt))
        # Power-preserving window normalization (matches IrtoolsLight)
        win = win * (jnp.sum(win**2) * Nt) / jnp.sum(win) ** 2
    else:
        win = jnp.ones(Nt)

    # Vectorised over spatial dimensions (no Python loop)
    x = dipoles.T  # (dim, Nt)
    x = x - jnp.mean(x, axis=1, keepdims=True)  # Remove DC component
    x = x * win[None, :]  # Apply window
    # Build mirrored, zero-padded signal: [x, zeros, flip(x)]
    pad = jnp.zeros((dim, 2 * Nzero))
    x_padded = jnp.concatenate([x, pad, jnp.flip(x, axis=1)], axis=1)

    # FFT scaled by dt (matches IrtoolsLight.psd)
    xFt = jnp.real(jnp.fft.fft(x_padded, axis=1))[:, : N // 2] * dt_fs
    spec = xFt**2  # Power spectrum (dim, N//2)

    # Intensity weighting: omega^2 / (dt * Nt * 2) * (2*pi)^2 / 2
    # dt kept in fs to match IrtoolsLight (consistent with dt_fs scaling above)
    spec = spec * (wf**2 / (dt_fs * Nt * 2) * (2 * jnp.pi) ** 2 / 2)[None, :]

    # Convert frequency to cm^-1: wf is in 1/fs = PHz
    PHztocm = 1e15 / C_LIGHT_CMS
    frequencies = wf * PHztocm
    intensities = jnp.sum(spec, axis=0)  # Sum over x, y, z

    if normalize:
        intensities = intensities / (jnp.max(intensities) + 1e-10)
    return frequencies, intensities


def _average_ir_results(frequencies_list, intensities_list, normalize=True):
    """Average IR spectra from multiple independent runs.

    All runs must share the same frequency axis (same ``dt_fs`` and number of
    NVE frames).  The averaged intensities are re-normalised so that the
    global peak equals 1 (unless normalize=False).

    Returns:
        frequencies : array – common frequency axis (cm⁻¹)
        mean_intensities : array – peak-normalised mean spectrum
        std_intensities  : array – standard deviation across runs
    """
    freq = jnp.asarray(frequencies_list[0])
    stacked = jnp.stack([jnp.asarray(i) for i in intensities_list])  # (n_runs, n_freq)
    mean_intens = jnp.mean(stacked, axis=0)
    if normalize:
        mean_intens = mean_intens / (jnp.max(mean_intens) + 1e-10)
    std_intens = jnp.std(stacked, axis=0)
    return freq, mean_intens, std_intens


def compute_ir_from_md_parallel(
    xyz_file: str,
    potential_name: str = "so3lr",
    model_path: Optional[str] = None,
    temperature: float = 300.0,
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    use_float64: bool = False,
    conformer_indices: Optional[List[int]] = None,
    n_replicas: int = 1,
    nvt_time_ps: float = 5.0,
    nve_time_ps: float = 25.0,
    dt_fs: float = 0.5,
    dipole_save_fs: float = 2.5,
    chop: Optional[int] = 10,
    window: str = "hann",
    parallel: bool = True,
    output_prefix: Optional[str] = None,
    potential_options: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
) -> Dict:
    """Compute IR spectrum from MD for multiple conformers and/or replicas.

    When ``parallel=True`` and ``n_total > 1`` (where
    ``n_total = len(conformer_indices) * n_replicas``), all NVT + NVE
    trajectories are run simultaneously with ``jax.vmap``.  The resulting
    per-trajectory IR spectra are averaged to give a single spectrum with
    improved statistics.

    Args:
        xyz_file: Path to input XYZ file.
        conformer_indices: List of 0-based conformer indices to process.
            ``None`` means use all conformers in the file.
        n_replicas: Number of independent MD replicas per conformer
            (different random seeds).  Default: 1.
        parallel: When ``True`` (default) and ``n_total > 1``, run all
            trajectories in a single vmapped kernel.  When ``False`` or
            ``n_total == 1``, run sequentially.
        atom_indices: Optional 0-based atom subset used to track the
            dipole (same semantics as :func:`compute_ir_from_md`).

    Returns:
        results dict compatible with :func:`save_ir_spectrum`, with extra
        keys ``n_replicas``, ``n_conformers``, ``n_total_trajectories``,
        and ``intensities_error`` (std across trajectories).
    """
    from jax_md import simulate, space, units

    from .sampling import _allocate_nbrs_batch

    unit = units.metal_unit_system()

    log_header("PARALLEL MD-BASED IR SPECTRUM CALCULATION")
    log_info(f"Loading structures from {xyz_file}...")

    structures = load_multiframe_xyz(xyz_file)
    n_conformers_total = len(structures)

    # Resolve which conformers to process
    if conformer_indices is None:
        conformer_indices = list(range(n_conformers_total))
        log_info(f"Processing all {n_conformers_total} conformers")
    else:
        for idx in conformer_indices:
            if idx >= n_conformers_total:
                raise ValueError(
                    f"Conformer index {idx} out of range (total: {n_conformers_total})"
                )
        log_info(f"Processing conformers: {conformer_indices}")

    n_confs = len(conformer_indices)
    n_total = n_confs * n_replicas
    log_info(f"Total trajectories: {n_confs} conformer(s) × {n_replicas} replica(s) = {n_total}")

    # Atom info from first conformer (all must share the same species)
    ref_structure = structures[conformer_indices[0]]
    numbers = ref_structure["numbers"]
    symbols = ref_structure["symbols"]
    masses_arr = get_atomic_masses(numbers)

    if use_float64:
        log_info("Using float64 precision")
        enable_float64()
        dtype = jnp.float64
    else:
        log_info("Using float32 precision")
        dtype = jnp.float32

    masses = jnp.array(masses_arr, dtype=dtype)

    n_atoms_batch = len(symbols)
    atom_indices = _normalize_atom_indices(atom_indices, n_atoms_batch)
    if atom_indices is not None:
        log_info(
            f"Tracking dipole over {len(atom_indices)} of {n_atoms_batch} atoms "
            f"(indices {atom_indices[0]}–{atom_indices[-1]})"
        )
        dipole_atoms_idx = jnp.asarray(atom_indices, dtype=jnp.int32)
    else:
        dipole_atoms_idx = None

    # ------------------------------------------------------------------
    # Sequential path: call compute_ir_from_md for each (conformer, replica)
    # ------------------------------------------------------------------
    if not parallel or n_total == 1:
        log_info(f"Running {n_total} trajectory/trajectories sequentially")
        all_freqs, all_intens = [], []
        first_result = None

        for c_idx, conf_i in enumerate(conformer_indices):
            for rep_i in range(n_replicas):
                run_id = c_idx * n_replicas + rep_i
                log_info(f"\nRun {run_id + 1}/{n_total}: conformer {conf_i}, replica {rep_i}")
                res = compute_ir_from_md(
                    xyz_file,
                    potential_name=potential_name,
                    model_path=model_path,
                    temperature=temperature,
                    charge=charge,
                    lr_cutoff=lr_cutoff,
                    use_float64=use_float64,
                    conformer_index=conf_i,
                    nvt_time_ps=nvt_time_ps,
                    nve_time_ps=nve_time_ps,
                    dt_fs=dt_fs,
                    dipole_save_fs=dipole_save_fs,
                    chop=chop,
                    window=window,
                    random_seed=rep_i,
                    potential_options=potential_options,
                    atom_indices=atom_indices,
                )
                all_freqs.append(res["frequencies"])
                all_intens.append(res["intensities"])
                if first_result is None:
                    first_result = res

        avg_freq, avg_intens, std_intens = _average_ir_results(all_freqs, all_intens)
        result = dict(first_result)
        result.update(
            {
                "frequencies": avg_freq,
                "intensities": avg_intens,
                "intensities_normalized": avg_intens,
                "intensities_error": std_intens,
                "n_replicas": n_replicas,
                "n_conformers": n_confs,
                "n_total_trajectories": n_total,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Parallel vmap path
    # ------------------------------------------------------------------
    log_info(f"Running {n_total} trajectories in parallel using vmap")

    # Build starting positions for each (conformer × replica) combination
    # Order: conf0_rep0, conf0_rep1, ..., conf0_repN, conf1_rep0, ...
    positions_list = []
    for conf_i in conformer_indices:
        pos = jnp.array(structures[conf_i]["positions"], dtype=dtype)
        for _ in range(n_replicas):
            positions_list.append(pos)
    positions_batch = jnp.stack(positions_list)  # (n_total, n_atoms, 3)

    # Build potential with partial charges
    log_info(f"Initializing {potential_name} potential with partial charges...")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        potential = _build_ir_potential(
            potential_name,
            numbers,
            lr_cutoff,
            charge,
            model_path,
            dtype,
            compute_charges=True,
            potential_options=potential_options,
        )
        potential.initialize(positions_batch[0])
        pot_result = potential.build_energy_fn(with_charges=True)

    if isinstance(pot_result, tuple):
        energy_fn, charges_fn = pot_result
    else:
        energy_fn, charges_fn = pot_result, None

    if charges_fn is None:
        raise RuntimeError(
            f"MD-based IR requires partial charges but the '{potential_name}' "
            "potential does not support them. Use a potential that provides "
            "partial charges (e.g. so3lr), or use Hessian-based IR instead."
        )

    # Allocate batch-normalised neighbor lists
    log_info("Allocating neighbor lists for all replicas...")
    nbr_state = _allocate_nbrs_batch(positions_batch, potential)

    # Helper: check if any neighbor list overflowed
    def _check_overflow(nbrs):
        if isinstance(nbrs, tuple):
            return nbrs[0].did_buffer_overflow.any() or nbrs[1].did_buffer_overflow.any()
        return nbrs.did_buffer_overflow.any()

    # MD parameters
    dt_ps = dt_fs / 1000.0
    dt = dt_ps * unit["time"]
    kT = temperature * unit["temperature"]
    gamma = 0.01

    nvt_steps = int(nvt_time_ps / dt_ps)
    nvt_save_interval = max(100, nvt_steps // 100)
    n_nvt_frames = nvt_steps // nvt_save_interval
    nvt_steps = n_nvt_frames * nvt_save_interval

    save_every = max(1, int(dipole_save_fs / dt_fs))
    nve_steps = int(nve_time_ps / dt_ps)
    n_nve_frames = nve_steps // save_every
    nve_steps = n_nve_frames * save_every

    log_header("PARALLEL NVT EQUILIBRATION")
    log_info(
        f"Temperature: {temperature:.0f} K, {dt_ps * nvt_steps} ps, {nvt_steps} steps in {n_nvt_frames} chunks"
    )

    _, shift_fn = space.free()
    init_fn_nvt, apply_fn_nvt = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma)

    update_nbrs_vmap = jax.vmap(potential.update_neighbors)

    def _init_nvt_single(key, pos):
        return init_fn_nvt(key, pos, mass=masses, kT=kT * 2)

    def _apply_nvt_single(state, **kwargs):
        return apply_fn_nvt(state, **kwargs)

    vmapped_init_nvt = jax.vmap(_init_nvt_single)
    vmapped_apply_nvt = jax.vmap(_apply_nvt_single)

    @jax.jit
    def run_nvt_inner_batch(states, nbrs):
        def body(_, carry):
            ss, nbs = carry
            nbs = update_nbrs_vmap(ss.position, nbs)
            nbr_kwargs = potential.get_neighbor_kwargs(nbs)
            ss = vmapped_apply_nvt(ss, **nbr_kwargs)
            return (ss, nbs)

        return jax.lax.fori_loop(0, nvt_save_interval, body, (states, nbrs))

    # Seed each replica with a unique key
    keys = jax.random.split(jax.random.PRNGKey(0), n_total)
    state_batch = vmapped_init_nvt(keys, positions_batch)

    for _ in tqdm(range(n_nvt_frames), desc="NVT Equilibration (parallel)", unit="chunk"):
        state_batch, nbr_state = run_nvt_inner_batch(state_batch, nbr_state)
        if _check_overflow(nbr_state):
            tqdm.write("  Neighbor list overflow during NVT, reallocating...")
            nbr_state = _allocate_nbrs_batch(state_batch.position, potential)

    log_header("PARALLEL NVE PRODUCTION RUN")
    log_info(f"NVE time: {nve_time_ps:.3g} ps, {nve_steps} steps, {n_nve_frames} frames")

    _, apply_fn_nve = simulate.nve(energy_fn, shift_fn, dt)

    def _apply_nve_single(state, **kwargs):
        return apply_fn_nve(state, **kwargs)

    vmapped_apply_nve = jax.vmap(_apply_nve_single)
    vmapped_charges = jax.vmap(charges_fn)

    # Initialise NVE state from equilibrated NVT state
    state_nve_batch = vmapped_apply_nve(state_batch)

    @jax.jit
    def run_nve_step_batch(states, nbrs):
        """Run save_every NVE steps, return (final_state, updated_nbrs, dipoles)."""

        def body(_, carry):
            ss, nbs = carry
            nbs = update_nbrs_vmap(ss.position, nbs)
            nbr_kwargs = potential.get_neighbor_kwargs(nbs)
            ss = vmapped_apply_nve(ss, **nbr_kwargs)
            return (ss, nbs)

        states_final, nbrs_final = jax.lax.fori_loop(0, save_every, body, (states, nbrs))
        # Compute per-replica dipole moments at the final step
        nbs_up = update_nbrs_vmap(states_final.position, nbrs_final)
        nbr_kwargs = potential.get_neighbor_kwargs(nbs_up)
        # charges_batch: (n_total, n_atoms)
        charges_batch = vmapped_charges(states_final.position, **nbr_kwargs)
        # μ = Σ q_i * (r_i − R_com). Restrict the sum (and the COM
        # reference) to the selected atoms when ``dipole_atoms_idx`` is
        # set so that only the chosen subsystem's dipole is tracked.
        if dipole_atoms_idx is not None:
            pos_sel = states_final.position[:, dipole_atoms_idx, :]
            q_sel = charges_batch[:, dipole_atoms_idx]
            m_sel = masses[dipole_atoms_idx]
            r_com = jnp.sum(m_sel[None, :, None] * pos_sel, axis=1) / jnp.sum(m_sel)
            pos_rel = pos_sel - r_com[:, None, :]
            dipoles = jnp.sum(q_sel[:, :, None] * pos_rel, axis=1)
        else:
            r_com = jnp.sum(masses[None, :, None] * states_final.position, axis=1) / jnp.sum(masses)
            pos_rel = states_final.position - r_com[:, None, :]
            dipoles = jnp.sum(charges_batch[:, :, None] * pos_rel, axis=1)
        return states_final, nbs_up, dipoles

    # Collect one dipole vector per replica per outer frame
    all_dipole_frames = []  # list of (n_total, 3) arrays

    for _ in tqdm(range(n_nve_frames), desc="NVE Production (parallel)", unit="frame"):
        state_nve_batch, nbr_state, dipoles_frame = run_nve_step_batch(state_nve_batch, nbr_state)
        all_dipole_frames.append(dipoles_frame)
        if _check_overflow(nbr_state):
            tqdm.write("  Neighbor list overflow during NVE, reallocating...")
            nbr_state = _allocate_nbrs_batch(state_nve_batch.position, potential)

    # all_dipole_frames: list[n_nve_frames] of (n_total, 3)
    # → stack → (n_nve_frames, n_total, 3) → transpose → (n_total, n_nve_frames, 3)
    dipoles_stack = jnp.stack(all_dipole_frames)  # (n_nve_frames, n_total, 3)
    dipoles_per_traj = jnp.transpose(dipoles_stack, (1, 0, 2))  # (n_total, n_nve_frames, 3)

    log_header("COMPUTING IR SPECTRA FROM DIPOLE AUTOCORRELATIONS")

    n_chop = chop if chop is not None and chop > 1 else 1
    if n_chop > 1:
        log_info(f"Applying Welch-style chopping: {n_chop} segments per trajectory")

    all_freqs, all_intens = [], []
    for i in range(n_total):
        dip = dipoles_per_traj[i]  # (n_nve_frames, 3)
        n_frames_i = dip.shape[0]

        if n_chop > 1:
            chunk_len = n_frames_i // n_chop
            chunk_freqs, chunk_intens = [], []
            for c in range(n_chop):
                start = c * chunk_len
                end = start + chunk_len
                f, inten = ir_from_dipoles(
                    dip[start:end], dt_fs=dipole_save_fs, window=window, normalize=False
                )
                chunk_freqs.append(f)
                chunk_intens.append(inten)
            avg_f, avg_i, _ = _average_ir_results(chunk_freqs, chunk_intens, normalize=False)
            all_freqs.append(avg_f)
            all_intens.append(avg_i)
        else:
            freqs, intens = ir_from_dipoles(
                dip, dt_fs=dipole_save_fs, window=window, normalize=False
            )
            all_freqs.append(freqs)
            all_intens.append(intens)

    # Normalize only at the very end (matching GEMS protocol)
    avg_freq, avg_intens, std_intens = _average_ir_results(all_freqs, all_intens, normalize=True)

    log_info(f"Averaged {n_total} spectra")
    log_info(f"Frequency range: 0 to {float(avg_freq[-1]):.0f} cm⁻¹")

    # Representative energy from first trajectory
    nbr_kwargs_final = potential.get_neighbor_kwargs(nbr_state)
    final_energies = jax.vmap(energy_fn)(state_nve_batch.position, **nbr_kwargs_final)
    mean_energy = float(jnp.mean(final_energies))

    results = {
        "frequencies": avg_freq,
        "intensities": avg_intens,
        "intensities_normalized": avg_intens,
        "intensities_error": std_intens,
        "temperature": temperature,
        "energy": mean_energy,
        "symbols": symbols,
        "positions": np.array(positions_batch[0]),
        "nvt_time_ps": nvt_time_ps,
        "nve_time_ps": nve_time_ps,
        "dt_fs": dt_fs,
        "save_interval": save_every,
        "n_replicas": n_replicas,
        "n_conformers": n_confs,
        "n_total_trajectories": n_total,
    }
    return results


# ============================================================================
# Plotting and output functions
# ============================================================================


def save_normal_mode_xyz(results: Dict, output_file: str = "normal_modes.xyz"):
    """Save all normal modes in a single multi-frame XYZ file.

    Creates a multi-frame XYZ file where each frame contains:
    - Equilibrium positions and displacement vectors in format: symbol x y z vx vy vz
    - Comment line with: Mode number, Frequency (cm⁻¹), Force Constant (eV/Å²)

    All modes (including translation/rotation) are saved.

    Args:
        results: Dictionary from compute_ir_spectrum() containing normal modes
        output_file: Output filename (default: "normal_modes.xyz")

    Returns:
        Output filename
    """
    # Check if this is Hessian-based (has 'normal_modes' key)
    if "normal_modes" not in results:
        log_info(
            "Warning: No normal mode data found. This function only works for Hessian-based IR."
        )
        return None

    frequencies = results["frequencies"]
    normal_modes = results["normal_modes"]
    positions = results["positions"]
    symbols = results["symbols"]
    masses = results["masses"]
    energy = results["energy"]
    temperature = results["temperature"]

    # Compute force constants
    c_cm_per_s = C_LIGHT * 100
    freq_hz = frequencies * c_cm_per_s
    omega = 2 * np.pi * freq_hz
    conversion = EV_TO_J / (ANGSTROM_TO_M**2 * AMU_TO_KG)
    force_constants = (omega**2) / conversion

    n_modes = len(frequencies)
    log_info(f"Saving all {n_modes} normal modes to {output_file}...")

    # Save all modes in a single multi-frame XYZ file
    with open(output_file, "w") as f:
        for mode_idx in range(n_modes):
            freq = frequencies[mode_idx]
            fc = force_constants[mode_idx]
            mode_vector = normal_modes[:, mode_idx]

            # Un-mass-weight the normal mode to get Cartesian displacement vectors
            # L_k[iα] / sqrt(m_i) gives the actual atomic displacement direction
            mass_inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
            cartesian_displacement = mode_vector * mass_inv_sqrt

            # Reshape to (n_atoms, 3)
            displacement_3d = cartesian_displacement.reshape(-1, 3)

            # Normalize displacement vector for visualization
            disp_norm = np.linalg.norm(displacement_3d)
            if disp_norm > 1e-10:
                displacement_3d = displacement_3d / disp_norm
            else:
                # Even if zero, still save it
                displacement_3d = displacement_3d * 0.0

            # Write frame for this mode
            f.write(f"{len(symbols)}\n")
            f.write(f"Mode {mode_idx}, Freq: {freq:.4f} cm⁻¹, Force Const: {fc:.6e} eV/Å²\n")

            for symbol, pos, disp in zip(symbols, positions, displacement_3d):
                # Write: symbol x y z vx vy vz
                f.write(f"{symbol:2s}  {pos[0]:>15.8f}  {pos[1]:>15.8f}  {pos[2]:>15.8f}  ")
                f.write(f"{disp[0]:>15.8f}  {disp[1]:>15.8f}  {disp[2]:>15.8f}\n")

    log_info(f"Saved all {n_modes} normal modes (including trans/rot) to {output_file}")

    return output_file


def save_optimized_structure(results: Dict, output_file: str = "optimized_structure.xyz"):
    """Save the optimized structure to XYZ file.

    This function saves the geometry-optimized structure used for Hessian calculation.

    Args:
        results: Dictionary from compute_ir_spectrum() containing optimized positions
        output_file: Output filename for the optimized structure (XYZ format)
    """
    from .utils import create_structure, save_structure

    # Create structure dictionary from results
    structure = create_structure(
        positions=jnp.array(results["positions"]), symbols=results["symbols"]
    )

    # Create comment with energy and temperature
    comment = f"Optimized structure, Energy: {results['energy']:.6f} eV, Temperature: {results['temperature']:.1f} K"

    # Use the utility function to save
    save_structure(output_file, structure, comment=comment)
    log_info(f"Optimized structure saved to {output_file}")


def save_ir_hessian_data(results: Dict, output_file: str = "ir_hessian_data.txt"):
    """Save detailed Hessian data including frequencies, force constants, and Hessian matrix.

    This function is only applicable for Hessian-based IR calculations.

    Args:
        results: Dictionary from compute_ir_spectrum() containing Hessian data
        output_file: Output filename for the detailed Hessian data
    """
    # Check if this is Hessian-based (has 'hessian' key)
    if "hessian" not in results:
        log_info("Warning: No Hessian data found. This function only works for Hessian-based IR.")
        return

    frequencies = results["frequencies"]
    hessian = results["hessian"]
    masses = results["masses"]
    symbols = results["symbols"]
    normal_modes = results["normal_modes"]

    # Compute force constants from frequencies
    # k = (2πν)² × μ where ν is in Hz
    # First convert cm⁻¹ to Hz: ν_Hz = ν_cm⁻¹ × c (in cm/s)
    c_cm_per_s = C_LIGHT * 100  # m/s to cm/s
    freq_hz = frequencies * c_cm_per_s  # Hz
    omega = 2 * np.pi * freq_hz  # rad/s

    # Force constants in eV/Angstrom² (from mass-weighted eigenvalues)
    # eigenvalue = ω²/conversion_factor where conversion_factor converts units
    conversion = EV_TO_J / (ANGSTROM_TO_M**2 * AMU_TO_KG)
    force_constants = (omega**2) / conversion  # eV/Angstrom²

    n_modes = len(frequencies)
    vib_threshold = 50.0  # cm⁻¹

    with open(output_file, "w") as f:
        f.write("# IR Hessian Data computed with MARS + SO3LR\n")
        f.write("#" + "=" * 78 + "\n")
        f.write(f"# Temperature: {results['temperature']:.1f} K\n")
        f.write(f"# Energy: {results['energy']:.6f} eV\n")
        f.write(f"# Number of atoms: {len(symbols)}\n")
        f.write(f"# Number of modes: {n_modes}\n")
        f.write("#" + "=" * 78 + "\n")
        f.write("\n")

        # Section 1: Frequencies and Force Constants
        f.write("# SECTION 1: VIBRATIONAL FREQUENCIES AND FORCE CONSTANTS\n")
        f.write("#" + "-" * 78 + "\n")
        f.write("# Mode    Frequency (cm⁻¹)    Force Constant (eV/Å²)    Type\n")
        f.write("#" + "-" * 78 + "\n")

        for i, (freq, fc) in enumerate(zip(frequencies, force_constants)):
            mode_type = "Vibrational" if abs(freq) > vib_threshold else "Trans/Rot"
            f.write(f"{i:>6d}    {freq:>16.4f}    {fc:>22.8e}    {mode_type}\n")

        f.write("\n\n")

        # Section 2: Hessian Matrix
        f.write("# SECTION 2: HESSIAN MATRIX (eV/Å²)\n")
        f.write("#" + "-" * 78 + "\n")
        f.write(f"# Shape: ({hessian.shape[0]}, {hessian.shape[1]})\n")
        f.write("# Format: 3N × 3N matrix where N is the number of atoms\n")
        f.write("# Indices correspond to: [atom_0_x, atom_0_y, atom_0_z, atom_1_x, ...]\n")
        f.write("#" + "-" * 78 + "\n")

        # Write atom mapping
        f.write("# Atom mapping:\n")
        for i, symbol in enumerate(symbols):
            f.write(f"#   Indices {3*i:3d}-{3*i+2:3d}: {symbol} (atom {i})\n")
        f.write("\n")

        # Write Hessian matrix in blocks for readability
        block_size = 6  # Number of columns per block
        n_blocks = (hessian.shape[1] + block_size - 1) // block_size

        for block in range(n_blocks):
            start_col = block * block_size
            end_col = min((block + 1) * block_size, hessian.shape[1])

            # Column headers
            f.write(f"# Columns {start_col} to {end_col-1}:\n")
            for i in range(hessian.shape[0]):
                for j in range(start_col, end_col):
                    f.write(f"{hessian[i, j]:>18.10e} ")
                f.write("\n")
            f.write("\n")

        f.write("\n")

        # Section 3: Normal Mode Eigenvectors
        f.write("# SECTION 3: NORMAL MODE EIGENVECTORS (Mass-weighted)\n")
        f.write("#" + "-" * 78 + "\n")
        f.write(f"# Shape: ({normal_modes.shape[0]}, {normal_modes.shape[1]})\n")
        f.write("# Each column is a normal mode eigenvector\n")
        f.write("# Format: 3N rows (atomic displacements) × 3N columns (modes)\n")
        f.write("#" + "-" * 78 + "\n")
        f.write("\n")

        # Write normal modes in blocks
        for block in range(n_blocks):
            start_col = block * block_size
            end_col = min((block + 1) * block_size, normal_modes.shape[1])

            f.write(f"# Modes {start_col} to {end_col-1}:\n")
            f.write(f"# Frequencies (cm⁻¹): ")
            for j in range(start_col, end_col):
                f.write(f"{frequencies[j]:>18.4f} ")
            f.write("\n")

            for i in range(normal_modes.shape[0]):
                for j in range(start_col, end_col):
                    f.write(f"{normal_modes[i, j]:>18.10e} ")
                f.write("\n")
            f.write("\n")

    log_info(f"Detailed Hessian data saved to {output_file}")


def save_mode_analysis(results: Dict, output_file: str = "mode_analysis.txt") -> str:
    """Save detailed vibrational mode classification analysis.

    Args:
        results: Dictionary from compute_ir_spectrum() with mode_classification
        output_file: Output filename

    Returns:
        Output filename
    """
    classification = results.get("mode_classification")
    if classification is None:
        if "normal_modes" not in results:
            log_info("Warning: No normal mode data. Mode analysis requires Hessian-based IR.")
            return None
        classification = classify_vibrational_modes(results)

    topology = classification["topology"]
    modes = classification["modes"]
    symbols = results["symbols"]
    n_atoms = len(symbols)
    frequencies = np.asarray(results["frequencies"])
    intensities = np.asarray(results["intensities_normalized"])
    n_total = len(frequencies)
    vib_threshold = 50.0  # cm⁻¹

    # Build lookup for classified vibrational modes
    mode_lookup = {}
    for m in modes:
        mode_lookup[m["mode_index"]] = m

    is_multi = classification.get("is_multimolecular", False)
    fragments = classification.get("fragments")

    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VIBRATIONAL MODE ANALYSIS\n")
        f.write("=" * 70 + "\n")
        if is_multi:
            f.write(
                f"System: {n_atoms} atoms, " f"{classification.get('n_fragments', 1)} molecules\n"
            )
        else:
            f.write(f"Molecule: {''.join(symbols)} ({n_atoms} atoms)\n")
        f.write(f"Temperature: {results['temperature']:.1f} K\n")
        f.write(f"Energy: {results['energy']:.6f} eV\n")
        f.write(f"Total modes: {n_total} (3N)\n")
        f.write("\n")

        # Molecular fragments (multi-molecule only)
        if is_multi and fragments:
            f.write("-" * 70 + "\n")
            f.write("MOLECULAR FRAGMENTS\n")
            f.write("-" * 70 + "\n")
            for fid, frag in enumerate(fragments):
                frag_syms = [symbols[a] for a in frag]
                # Build molecular formula
                from collections import Counter

                elem_counts = Counter(frag_syms)
                # Hill order: C first, H second, then alphabetical
                hill = []
                for el in ("C", "H"):
                    if el in elem_counts:
                        hill.append(el + (str(elem_counts.pop(el)) if elem_counts[el] > 1 else ""))
                        del elem_counts[el]
                for el in sorted(elem_counts):
                    hill.append(el + (str(elem_counts[el]) if elem_counts[el] > 1 else ""))
                formula = "".join(hill)
                f.write(
                    f"  mol {fid}: {formula} "
                    f"({len(frag)} atoms, indices {frag[0]}-{frag[-1]})\n"
                )
            f.write("\n")

        # Topology summary
        bond_orders = topology.get("bond_orders", {})
        n_single = sum(1 for o in bond_orders.values() if o == 1)
        n_double = sum(1 for o in bond_orders.values() if o == 2)
        n_triple = sum(1 for o in bond_orders.values() if o == 3)

        f.write("-" * 70 + "\n")
        f.write("MOLECULAR TOPOLOGY\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Bonds:     {len(topology['bonds'])} "
            f"({n_single} single, {n_double} double, {n_triple} triple)\n"
        )
        f.write(f"Angles:    {len(topology['angles'])}\n")
        f.write(f"Dihedrals: {len(topology['dihedrals'])}\n")
        f.write("\n")

        # List bonds with bond order
        order_names = {1: "single", 2: "double", 3: "triple"}
        f.write("Bonds:\n")
        for idx, (i, j) in enumerate(topology["bonds"]):
            dist = topology["bond_distances"][idx]
            key = (min(i, j), max(i, j))
            order = bond_orders.get(key, 1)
            sym = BOND_ORDER_SYMBOLS.get(order, "-")
            oname = order_names.get(order, "single")
            f.write(f"  {symbols[i]}({i}){sym}{symbols[j]}({j})  " f"d={dist:.3f} A  [{oname}]\n")
        f.write("\n")

        # Classification summary
        n_transrot = sum(1 for freq in frequencies if abs(freq) < vib_threshold)
        f.write("-" * 70 + "\n")
        f.write("MODE CLASSIFICATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Trans/Rot:  {n_transrot}\n")
        f.write(f"Stretching: {classification['n_stretching']}\n")
        f.write(f"Bending:    {classification['n_bending']}\n")
        f.write(f"Torsion:    {classification['n_torsion']}\n")
        f.write(f"Mixed:      {classification['n_mixed']}\n")
        if is_multi:
            f.write(f"Intermolecular: {classification.get('n_intermolecular', 0)}\n")
        total_vib = (
            classification["n_stretching"]
            + classification["n_bending"]
            + classification["n_torsion"]
            + classification["n_mixed"]
        )
        f.write(f"Total vibrational: {total_vib}\n")
        f.write(f"Total modes: {n_total}\n")
        f.write("\n")

        # All modes table (translation/rotation + vibrational)
        f.write("-" * 100 + "\n")
        f.write("ALL MODES\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Mode':<6} {'Freq (cm-1)':<14} {'Intensity':<12} {'Type':<12} "
            f"{'Label':<30} {'Functional Group'}\n"
        )
        f.write("-" * 100 + "\n")

        sorted_indices = np.argsort(frequencies)
        for idx in sorted_indices:
            freq = frequencies[idx]
            intensity = intensities[idx]
            m_info = mode_lookup.get(int(idx))

            if m_info:
                fg = m_info.get("functional_group", "")
                f.write(
                    f"{idx:<6} {freq:>12.1f}   "
                    f"{intensity:>10.4f}   {m_info['type']:<12} "
                    f"{m_info['label']:<30} {fg}\n"
                )
            else:
                mode_type = "trans/rot" if abs(freq) < vib_threshold else "unclassified"
                f.write(
                    f"{idx:<6} {freq:>12.1f}   "
                    f"{intensity:>10.4f}   {mode_type:<12} "
                    f"{'':<30}\n"
                )

        f.write("\n")

        # Detailed vibrational mode analysis
        f.write("-" * 100 + "\n")
        f.write("DETAILED VIBRATIONAL MODE ANALYSIS\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Mode':<6} {'Freq (cm-1)':<14} {'Intensity':<12} {'Type':<12} "
            f"{'Label':<30} {'Functional Group'}\n"
        )
        f.write("-" * 100 + "\n")

        for m in sorted(modes, key=lambda x: x["frequency"]):
            fg = m.get("functional_group", "")
            f.write(
                f"{m['mode_index']:<6} {m['frequency']:>12.1f}   "
                f"{m['intensity']:>10.4f}   {m['type']:<12} "
                f"{m['label']:<30} {fg}\n"
            )

            # Score breakdown
            s = m["scores"]
            f.write(
                f"{'':>6} Scores: stretch={s['stretching']:.3f}  "
                f"bend={s['bending']:.3f}  torsion={s['torsion']:.3f}"
            )
            if m["sub_type"]:
                f.write(f"  sub-type: {m['sub_type']}")
            f.write("\n")

            # Atoms involved
            atoms_str = ", ".join(f"{sym}({idx})" for idx, sym in m["dominant_atoms"])
            f.write(f"{'':>6} Atoms: {atoms_str}\n")

            # Fragment contributions (multi-molecule only)
            frag_contribs = m.get("fragment_contributions")
            if frag_contribs and is_multi:
                parts = [
                    f"mol {fid}: {frac:.1%}"
                    for fid, frac in sorted(frag_contribs.items())
                    if frac > 0.01
                ]
                frag_id = m.get("fragment_id")
                tag = "intermolecular" if frag_id == "collective" else f"mol {frag_id}"
                f.write(f"{'':>6} Fragment: {tag} ({', '.join(parts)})\n")

            f.write("\n")

        # Top intense modes section
        f.write("-" * 70 + "\n")
        f.write("TOP INTENSE MODES\n")
        f.write("-" * 70 + "\n")
        sorted_by_intensity = sorted(modes, key=lambda x: x["intensity"], reverse=True)
        for m in sorted_by_intensity[:20]:
            fg = m.get("functional_group", "")
            fg_str = f"  [{fg}]" if fg else ""
            f.write(
                f"  {m['frequency']:>8.1f} cm-1  I={m['intensity']:.4f}  " f"{m['label']}{fg_str}\n"
            )

    log_info(f"Mode analysis saved to {output_file}")
    return output_file


def save_ir_spectrum(results: Dict, output_file: str = "ir_spectrum.dat"):
    """Save IR spectrum to file.

    Args:
        results: Dictionary from compute_ir_spectrum() or compute_ir_from_md()
        output_file: Output filename
    """
    frequencies = results["frequencies"]
    intensities = results["intensities_normalized"]

    # Check if this is MD-based or Hessian-based
    is_md = "nvt_time_ps" in results

    with open(output_file, "w") as f:
        if is_md:
            f.write("# IR Spectrum computed with MARS + SO3LR (MD-based)\n")
            f.write(f"# Temperature: {results['temperature']:.1f} K\n")
            f.write(f"# Energy: {results['energy']:.6f} eV\n")
            f.write(f"# NVT time: {results['nvt_time_ps']:.1f} ps\n")
            f.write(f"# NVE time: {results['nve_time_ps']:.1f} ps\n")
            f.write(f"# Timestep: {results['dt_fs']:.3f} fs\n")
        else:
            f.write("# IR Spectrum computed with MARS + SO3LR (Analytical Hessian)\n")
            f.write(f"# Temperature: {results['temperature']:.1f} K\n")
            f.write(f"# Energy: {results['energy']:.6f} eV\n")
        f.write("#\n")
        f.write("# Frequency (cm⁻¹)    Intensity (normalized)\n")

        for freq, intensity in zip(frequencies, intensities):
            f.write(f"{freq:>16.2f}    {intensity:>18.6f}\n")

    log_info(f"IR spectrum saved to {output_file}")


def save_ir_plot(
    results: Dict,
    output_file: str = "ir_spectrum.png",
    freq_range: tuple = (400, 4000),
    broadening: float = 10.0,
    dpi: int = 300,
    annotate: bool = True,
    top_n_labels: int = 8,
    gaussian_sigma: float = 0.0,
):
    """Save IR spectrum plot as an image.

    For Hessian-based spectra with mode classification, the stick spectrum
    is color-coded by mode type and the top modes are labeled.

    Args:
        results: Dictionary from compute_ir_spectrum() or compute_ir_from_md()
        output_file: Output image filename (png, pdf, svg, etc.)
        freq_range: Tuple of (min_freq, max_freq) in cm⁻¹
        broadening: Lorentzian broadening width in cm⁻¹ (only for Hessian-based)
        dpi: Image resolution in dots per inch
        annotate: If True, color-code and label modes (requires mode_classification)
        top_n_labels: Number of top intense modes to label on stick spectrum
        gaussian_sigma: Standard deviation for Gaussian smoothing of MD spectrum
            (in number of data points). Set to 0 to disable. Only applies to MD-based IR.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_info("Warning: matplotlib not available. Skipping plot generation.")
        log_info("Install with: pip install matplotlib")
        return

    frequencies = results["frequencies"]
    intensities = results["intensities_normalized"]

    # Check if this is MD-based or Hessian-based
    is_md = "nvt_time_ps" in results

    # Filter vibrational modes
    if is_md:
        vib_mask = frequencies > 50.0
    else:
        vib_mask = np.abs(frequencies) > 50.0
    vib_indices = np.where(vib_mask)[0]
    frequencies = frequencies[vib_mask]
    intensities = intensities[vib_mask]

    if is_md:
        # MD-based: single plot with continuous spectrum (unchanged)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_plot = frequencies[mask]
        int_plot = intensities[mask]

        if gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter1d

            int_plot = gaussian_filter1d(int_plot, sigma=gaussian_sigma)

        ax.plot(freq_plot, int_plot, "b-", linewidth=1.5)
        ax.fill_between(freq_plot, int_plot, alpha=0.3)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Normalized Intensity", fontsize=12)
        ax.set_title("IR Spectrum from Molecular Dynamics", fontsize=14, fontweight="bold")
        ax.set_xlim(freq_range)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

        info_text = (
            f"MD-based IR\nT = {results['temperature']:.0f} K\nE = {results['energy']:.4f} eV\n"
        )
        info_text += f"NVT: {results['nvt_time_ps']:.0f} ps\nNVE: {results['nve_time_ps']:.0f} ps"
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

    else:
        # Hessian-based: two subplots (stick + broadened)
        classification = results.get("mode_classification") if annotate else None
        has_classification = classification is not None and len(classification.get("modes", [])) > 0

        # Build mode index -> classification lookup
        mode_lookup = {}
        if has_classification:
            for m in classification["modes"]:
                mode_lookup[m["mode_index"]] = m

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        # --- Stick spectrum (color-coded if classification available) ---
        if has_classification:
            # Plot each stick individually with color based on type
            for freq, intensity, orig_idx in zip(frequencies, intensities, vib_indices):
                m_info = mode_lookup.get(int(orig_idx))
                if m_info:
                    color = MODE_TYPE_COLORS.get(m_info["type"], "#888888")
                else:
                    color = "#888888"
                ax1.vlines(freq, 0, intensity, colors=color, linewidth=1.5)
                ax1.plot(freq, intensity, "o", color=color, markersize=3)

            # Label top N most intense modes
            mode_list = classification["modes"]
            in_range = [m for m in mode_list if freq_range[0] <= m["frequency"] <= freq_range[1]]
            top_modes = sorted(in_range, key=lambda x: x["intensity"], reverse=True)[:top_n_labels]

            for idx, m in enumerate(top_modes):
                freq_m = m["frequency"]
                int_m = m["intensity"]
                short_label = _peak_label(m)

                # Alternate annotation offset to avoid overlap
                y_offset = 0.05 + 0.04 * (idx % 3)
                ax1.annotate(
                    short_label,
                    xy=(freq_m, int_m),
                    xytext=(freq_m, int_m + y_offset),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    rotation=45,
                    color=MODE_TYPE_COLORS.get(m["type"], "#888888"),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )
        else:
            # Fallback: plain blue stick spectrum
            ax1.stem(frequencies, intensities, basefmt=" ", linefmt="b-", markerfmt="bo")

        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax1.set_ylabel("Normalized Intensity", fontsize=12)
        ax1.set_title("IR Stick Spectrum", fontsize=14, fontweight="bold")
        ax1.set_xlim(freq_range)
        ax1.set_ylim(0, 1.3)
        ax1.grid(True, alpha=0.3)

        # --- Broadened spectrum with Lorentzian lineshape ---
        freq_axis = np.linspace(freq_range[0], freq_range[1], int(freq_range[1] - freq_range[0]))
        spectrum = np.zeros_like(freq_axis)

        for freq, intensity in zip(frequencies, intensities):
            if freq_range[0] <= freq <= freq_range[1]:
                spectrum += intensity * broadening**2 / ((freq_axis - freq) ** 2 + broadening**2)

        ax2.plot(freq_axis, spectrum, "b-", linewidth=1.5)
        ax2.fill_between(freq_axis, spectrum, alpha=0.3)

        # Add colored markers at peak positions on broadened spectrum
        if has_classification:
            for freq, intensity, orig_idx in zip(frequencies, intensities, vib_indices):
                if freq_range[0] <= freq <= freq_range[1] and intensity > 0.05:
                    m_info = mode_lookup.get(int(orig_idx))
                    if m_info:
                        color = MODE_TYPE_COLORS.get(m_info["type"], "#888888")
                        # Find the broadened spectrum value at this frequency
                        idx_near = np.argmin(np.abs(freq_axis - freq))
                        ax2.plot(
                            freq, spectrum[idx_near], "v", color=color, markersize=5, alpha=0.8
                        )

        ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax2.set_ylabel("Intensity (arbitrary units)", fontsize=12)
        ax2.set_title(
            f"IR Spectrum (Lorentzian, \u03b3={broadening:.1f} cm\u207b\u00b9)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlim(freq_range)
        ax2.set_ylim(0, None)
        ax2.grid(True, alpha=0.3)

        # Add info text
        info_text = (
            f"Hessian-based IR\nT = {results['temperature']:.0f} K\nE = {results['energy']:.4f} eV"
        )
        if has_classification:
            info_text += (
                f"\nStr: {classification['n_stretching']}  "
                f"Bend: {classification['n_bending']}  "
                f"Tor: {classification['n_torsion']}  "
                f"Mix: {classification['n_mixed']}"
            )

        ax2.text(
            0.98,
            0.98,
            info_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    log_info(f"IR spectrum plot saved to {output_file}")


# ---------------------------------------------------------------------------
# Functional-group context recognition
# ---------------------------------------------------------------------------


def _get_functional_group_context(
    dominant_atoms: list, symbols: list, topology: dict, mode_type: str = ""
) -> str:
    """Return a chemically descriptive functional group label for a vibrational mode.

    Examines the bonding neighbourhood of the dominant atoms to identify the
    chemical context across all common organic (and some inorganic) functional
    groups (e.g. "C=O (ketone)", "O-H (alcohol)", "C-H (CH3, methyl)").
    If the pattern is not recognised a generic bond label is returned, just
    like the previous behaviour (e.g. "C-N", "S=O").

    Args:
        dominant_atoms: List of atom indices *or* (idx, symbol) pairs.
        symbols: Element symbol for every atom in the molecule.
        topology: Topology dict from ``_build_topology`` (needs
            ``neighbors`` and ``bond_orders``).
        mode_type: 'stretching', 'bending', 'torsion', or 'mixed'
            (unused here but kept for API compatibility).

    Returns:
        A human-readable functional-group string.
    """
    nbrs_map = topology.get("neighbors", {})
    bond_orders = topology.get("bond_orders", {})

    # Normalise dominant_atoms to plain integer indices
    atom_indices = []
    for entry in dominant_atoms:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            atom_indices.append(int(entry[0]))
        else:
            atom_indices.append(int(entry))

    # ── small helper functions ─────────────────────────────────────────────
    def sym(i):
        return symbols[i] if i < len(symbols) else "?"

    def bo(i, j):
        return bond_orders.get((min(i, j), max(i, j)), 1)

    def nbrs(i):
        return nbrs_map.get(i, set())

    def hn(i):
        """Heavy (non-H) neighbours of i."""
        return [n for n in nbrs(i) if sym(n) != "H"]

    def hc(i):
        """Number of H neighbours of i."""
        return sum(1 for n in nbrs(i) if sym(n) == "H")

    def has_bo(i, element, order):
        return any(bo(i, n) == order and sym(n) == element for n in nbrs(i))

    def is_in_ring(i):
        """True if i is in any ring (path-of-2 check, works for 3-6-membered)."""
        nl = list(nbrs(i))
        for n in nl:
            if nbrs(n) & set(nl) - {i}:
                return True
        return False

    def is_in_3ring(i):
        """True if i is in a 3-membered ring."""
        nl = list(nbrs(i))
        for a in range(len(nl)):
            for b in range(a + 1, len(nl)):
                if nl[b] in nbrs(nl[a]):
                    return True
        return False

    def is_sp2_ring(i):
        """True if i is in a ring AND has a double bond to a ring neighbour."""
        if not is_in_ring(i):
            return False
        return any(bo(i, n) == 2 for n in nbrs(i) if is_in_ring(n))

    def is_aromatic_C(i):
        return sym(i) == "C" and is_sp2_ring(i)

    def is_pyrrole_N(i):
        """Pyrrole-type N: in aromatic ring with N-H."""
        return sym(i) == "N" and is_sp2_ring(i) and hc(i) >= 1

    def has_heteroatom_ring_nbr(i):
        return any(sym(n) in ("N", "O", "S") for n in nbrs(i) if is_in_ring(n))

    # ── early exit: fewer than 2 atoms ────────────────────────────────────
    if len(atom_indices) < 2:
        return sym(atom_indices[0]) if atom_indices else "unclassified"

    i, j = atom_indices[0], atom_indices[1]
    si, sj = sym(i), sym(j)
    bo_ij = bo(i, j)
    ep = frozenset([si, sj])  # element-pair (order-independent)

    # =========================================================================
    # 1.  X-H bonds
    # =========================================================================
    h_idx = heavy_idx = None
    if si == "H":
        h_idx, heavy_idx = i, j
    elif sj == "H":
        h_idx, heavy_idx = j, i

    if heavy_idx is not None:
        sh = sym(heavy_idx)

        # ── O-H ──────────────────────────────────────────────────────────
        if sh == "O":
            hn_o = hn(heavy_idx)
            if any(sym(n) == "O" for n in hn_o):
                return "O-H (hydroperoxide)"
            if any(sym(n) == "N" for n in hn_o):
                return "O-H (hydroxamic acid / oxime)"
            if any(sym(n) == "P" for n in hn_o):
                return "O-H (phosphoric acid)"
            if any(sym(n) == "S" for n in hn_o):
                return "O-H (sulfonic acid)"
            if any(sym(n) == "B" for n in hn_o):
                return "O-H (boronic acid)"
            c_nbr = next((n for n in hn_o if sym(n) == "C"), None)
            if c_nbr is not None:
                if has_bo(c_nbr, "O", 2):
                    return "O-H (carboxylic acid)"
                if is_aromatic_C(c_nbr):
                    return "O-H (phenol)"
                if has_bo(c_nbr, "C", 2):
                    return "O-H (enol)"
            return "O-H (alcohol)"

        # ── N-H ──────────────────────────────────────────────────────────
        if sh == "N":
            hn_n = hn(heavy_idx)
            if is_pyrrole_N(heavy_idx):
                return "N-H (pyrrole/indole)"
            c_co = [n for n in hn_n if sym(n) == "C" and has_bo(n, "O", 2)]
            if c_co:
                c = c_co[0]
                n_count = sum(1 for n2 in hn(c) if sym(n2) == "N")
                has_co_o = any(sym(n2) == "O" and bo(c, n2) == 1 for n2 in hn(c))
                if n_count >= 2:
                    return "N-H (urea)"
                if has_co_o:
                    return "N-H (carbamate/urethane)"
                return "N-H (amide)"
            if any(sym(n) == "S" and has_bo(n, "O", 2) for n in hn_n):
                return "N-H (sulfonamide)"
            if any(sym(n) == "N" for n in hn_n):
                return "N-H (hydrazine/hydrazone)"
            cn_n = [n for n in hn_n if sym(n) == "C"]
            if cn_n and sum(1 for n2 in hn(cn_n[0]) if sym(n2) == "N") >= 2:
                return "N-H (guanidine/amidine)"
            if hc(heavy_idx) >= 2:
                return (
                    "N-H (aniline, primary)"
                    if any(is_aromatic_C(n) for n in hn_n)
                    else "N-H (primary amine)"
                )
            return "N-H (secondary amine)"

        # ── C-H ──────────────────────────────────────────────────────────
        if sh == "C":
            h_num = hc(heavy_idx)
            if has_bo(heavy_idx, "C", 3) or has_bo(heavy_idx, "N", 3):
                return "C-H (sp, alkynyl)"
            if is_aromatic_C(heavy_idx):
                return (
                    "C-H (heteroaromatic)"
                    if has_heteroatom_ring_nbr(heavy_idx)
                    else "C-H (aromatic)"
                )
            if has_bo(heavy_idx, "O", 2):
                return "C-H (aldehyde)"
            if has_bo(heavy_idx, "C", 2) or has_bo(heavy_idx, "N", 2):
                return "C-H (vinyl/sp2)"
            # sp3 context
            hn_c = hn(heavy_idx)
            adj = {sym(n) for n in hn_c}
            suffix = "CH%d" % h_num if h_num <= 3 else "CH"
            if any(has_bo(n, "O", 2) for n in hn_c):
                return "C-H (%s, alpha-carbonyl)" % suffix
            if any(has_bo(n, "S", 2) for n in hn_c):
                return "C-H (%s, alpha-thiocarbonyl)" % suffix
            if "O" in adj:
                return "C-H (%s, alpha-ether/alcohol)" % suffix
            if "N" in adj:
                return "C-H (%s, alpha-amine)" % suffix
            if "S" in adj:
                return "C-H (%s, alpha-thioether)" % suffix
            if adj & {"F", "Cl", "Br", "I"}:
                return "C-H (%s, alpha-halo)" % suffix
            if h_num == 3:
                return "C-H (CH3, methyl)"
            if h_num == 2:
                return (
                    "C-H (CH2, cyclopropane)" if is_in_3ring(heavy_idx) else "C-H (CH2, methylene)"
                )
            if h_num == 1:
                return "C-H (CH, methine)"
            return "C-H (sp3)"

        # ── S-H / P-H / Si-H / B-H ───────────────────────────────────────
        if sh == "S":
            return (
                "S-H (thiophenol)"
                if any(is_aromatic_C(n) for n in hn(heavy_idx))
                else "S-H (thiol)"
            )
        if sh == "P":
            return "P-H (phosphine)"
        if sh == "Si":
            return "Si-H (silane)"
        if sh == "B":
            return "B-H (borane)"

        # Generic X-H fallback  (preserves old atom-name behaviour)
        return "%s-H" % sh

    # =========================================================================
    # 2.  Heavy atom - Heavy atom bonds
    # =========================================================================

    # ── C=O (carbonyl family) ─────────────────────────────────────────────
    if ep == frozenset(["C", "O"]) and bo_ij == 2:
        c_idx = i if si == "C" else j
        hn_c = hn(c_idx)
        if hc(c_idx) >= 1:
            return "C=O (aldehyde)"
        o_single = [n for n in hn_c if sym(n) == "O" and bo(c_idx, n) == 1]
        n_adj = [n for n in hn_c if sym(n) == "N"]
        hal_adj = [n for n in hn_c if sym(n) in ("F", "Cl", "Br", "I")]
        s_adj = [n for n in hn_c if sym(n) == "S"]
        if o_single:
            o2 = o_single[0]
            if any(has_bo(n2, "O", 2) for n2 in hn(o2) if n2 != c_idx):
                return "C=O (anhydride)"
            if hc(o2) >= 1:
                return "C=O (carboxylic acid)"
            if len(o_single) >= 2:
                return "C=O (carbonate)"
            if n_adj:
                return "C=O (carbamate/urethane)"
            return "C=O (lactone)" if is_in_ring(c_idx) else "C=O (ester)"
        if hal_adj:
            return "C=O (acyl halide)"
        if n_adj:
            if len(n_adj) >= 2:
                return "C=O (urea)"
            return "C=O (lactam)" if is_in_ring(c_idx) else "C=O (amide)"
        if s_adj:
            return "C=O (thioester)"
        return "C=O (ketone)"

    # ── C=S (thiocarbonyl family) ─────────────────────────────────────────
    if ep == frozenset(["C", "S"]) and bo_ij == 2:
        c_idx = i if si == "C" else j
        return (
            "C=S (thioamide)"
            if any(sym(n) == "N" for n in hn(c_idx))
            else "C=S (thiocarbonyl/thioketone)"
        )

    # ── C=C (double bond family) ──────────────────────────────────────────
    if ep == frozenset(["C", "C"]) and bo_ij == 2:
        if is_aromatic_C(i) or is_aromatic_C(j):
            ring_adj = {n for n in list(nbrs(i)) + list(nbrs(j)) if is_in_ring(n)}
            if any(sym(n) in ("N", "O", "S") for n in ring_adj):
                return "C=C (heteroaromatic)"
            return "C=C (aromatic)"
        if has_bo(i, "C", 2) and has_bo(j, "C", 2):
            return "C=C (allene/cumulene)"
        if any(sym(n) == "N" for n in hn(i) + hn(j)):
            return "C=C (enamine)"
        if any(sym(n) == "O" for n in hn(i) + hn(j)):
            return "C=C (enol/vinyl ether)"
        return "C=C (alkene)"

    # ── C=N (imine family) ───────────────────────────────────────────────
    if ep == frozenset(["C", "N"]) and bo_ij == 2:
        n_idx = i if si == "N" else j
        hn_n = hn(n_idx)
        if any(sym(n) == "O" and hc(n) >= 1 for n in nbrs(n_idx)):
            return "C=N (oxime)"
        if any(sym(n) == "N" for n in hn_n):
            return "C=N (hydrazone)"
        if is_in_ring(n_idx):
            return "C=N (imine, ring)"
        return "C=N (imine/Schiff base)"

    # ── C-N het-cumulenes: check BEFORE the single-bond C-N block ────────
    # (frozenset C,N bo==2 already handled above, bo==1 below)

    # ── C triple bonds ────────────────────────────────────────────────────
    if ep == frozenset(["C", "N"]) and bo_ij == 3:
        return "C=N (nitrile)"
    if ep == frozenset(["C", "C"]) and bo_ij == 3:
        return "C=C (alkyne)"

    # ── N=N / N-N / N triple ────────────────────────────────────────────
    if si == "N" and sj == "N":
        if bo_ij == 3:
            return "N=N (diazonium)"
        if bo_ij == 2:
            return "N=N (azo)"
        return "N-N (hydrazine)"

    # ── N=O / N-O ────────────────────────────────────────────────────────
    if ep == frozenset(["N", "O"]) and bo_ij == 2:
        n_idx = i if si == "N" else j
        return (
            "N=O (nitro group)"
            if sum(1 for n in nbrs(n_idx) if sym(n) == "O") >= 2
            else "N=O (nitroso)"
        )
    if ep == frozenset(["N", "O"]) and bo_ij == 1:
        n_idx = i if si == "N" else j
        if any(has_bo(n2, "O", 2) for n2 in hn(n_idx)):
            return "N-O (N-oxide)"
        return "N-O (hydroxylamine)"

    # ── O-O ──────────────────────────────────────────────────────────────
    if si == "O" and sj == "O":
        return "O-O (hydroperoxide)" if hc(i) + hc(j) >= 1 else "O-O (peroxide)"

    # ── S-S ──────────────────────────────────────────────────────────────
    if si == "S" and sj == "S":
        return "S-S (disulfide)"

    # ── S=O ──────────────────────────────────────────────────────────────
    if ep == frozenset(["S", "O"]) and bo_ij == 2:
        s_idx = i if si == "S" else j
        n_so2 = sum(1 for n in nbrs(s_idx) if sym(n) == "O" and bo(s_idx, n) == 2)
        if n_so2 >= 2:
            if any(sym(n) == "O" and hc(n) >= 1 for n in hn(s_idx)):
                return "S=O (sulfonic acid)"
            if any(sym(n) == "N" for n in hn(s_idx)):
                return "S=O (sulfonamide)"
            return "S=O (sulfone)"
        return "S=O (sulfoxide)"

    # ── P=O / P-O ────────────────────────────────────────────────────────
    if ep == frozenset(["P", "O"]) and bo_ij == 2:
        p_idx = i if si == "P" else j
        if any(sym(n) == "N" for n in hn(p_idx)):
            return "P=O (phosphonamide)"
        if any(sym(n) == "O" and hc(n) >= 1 for n in hn(p_idx)):
            return "P=O (phosphonic acid)"
        return "P=O (phosphate/phosphonate)"
    if ep == frozenset(["P", "O"]) and bo_ij == 1:
        return "P-O (phosphate ester)"

    # ── C-O (single bond) ────────────────────────────────────────────────
    if ep == frozenset(["C", "O"]) and bo_ij == 1:
        o_idx = i if si == "O" else j
        c_idx = j if si == "O" else i
        if hc(o_idx) >= 1:
            return "C-O (phenol)" if is_aromatic_C(c_idx) else "C-O (alcohol)"
        o_hn = hn(o_idx)
        if len(o_hn) == 2:
            if is_in_3ring(o_idx):
                return "C-O (epoxide)"
            if any(sym(n) == "O" for n in o_hn):
                return "C-O (peroxide)"
            if has_bo(c_idx, "C", 2) or any(has_bo(n2, "C", 2) for n2 in o_hn if n2 != c_idx):
                return "C-O (vinyl ether/enol ether)"
            if has_bo(c_idx, "O", 2):
                return "C-O (ester)"
            return "C-O (ether)"
        if has_bo(c_idx, "O", 2):
            return "C-O (ester)"
        return "C-O (alcohol)"

    # ── C-N (single bond) ────────────────────────────────────────────────
    if ep == frozenset(["C", "N"]) and bo_ij == 1:
        n_idx = i if si == "N" else j
        c_idx = j if si == "N" else i
        hn_n = hn(n_idx)
        hc_n = hc(n_idx)
        if is_aromatic_C(c_idx) and hc_n >= 1:
            return "C-N (aniline/aromatic amine)"
        if any(has_bo(n2, "O", 2) for n2 in hn_n if sym(n2) == "C"):
            c2 = next(n2 for n2 in hn_n if sym(n2) == "C" and has_bo(n2, "O", 2))
            if any(sym(n3) == "O" and bo(c2, n3) == 1 for n3 in hn(c2)):
                return "C-N (carbamate)"
            return "C-N (amide)"
        if any(sym(n) == "S" and has_bo(n, "O", 2) for n in hn_n):
            return "C-N (sulfonamide)"
        if hc_n >= 2:
            return "C-N (primary amine)"
        if hc_n == 1:
            return "C-N (secondary amine)"
        return "C-N (tertiary amine)"

    # ── C-S (single bond) ────────────────────────────────────────────────
    if ep == frozenset(["C", "S"]) and bo_ij == 1:
        s_idx = i if si == "S" else j
        if hc(s_idx) >= 1:
            return "C-S (thiol)"
        if any(sym(n) == "S" for n in hn(s_idx)):
            return "C-S (disulfide)"
        if has_bo(s_idx, "O", 2):
            return "C-S (sulfoxide/sulfone)"
        if has_bo(s_idx, "C", 3):
            return "C-S (thiocyanate)"
        return "C-S (thioether)"

    # ── C-X (halogens) ───────────────────────────────────────────────────
    for hal in ("F", "Cl", "Br", "I"):
        if ep == frozenset(["C", hal]):
            c_idx = i if si == "C" else j
            if is_aromatic_C(c_idx):
                return "C-%s (aryl halide)" % hal
            if has_bo(c_idx, "C", 2):
                return "C-%s (vinyl halide)" % hal
            if has_bo(c_idx, "O", 2):
                return "C-%s (acyl halide)" % hal
            return "C-%s (alkyl halide)" % hal

    # ── Si, B, P ─────────────────────────────────────────────────────────
    if ep == frozenset(["Si", "O"]):
        return "Si-O (silyl ether/siloxane)"
    if ep == frozenset(["Si", "C"]):
        return "Si-C (organosilane)"
    if ep == frozenset(["B", "O"]):
        o_idx = i if si == "O" else j
        return "B-O (boronic acid)" if hc(o_idx) >= 1 else "B-O (boronate ester)"

    # ── C-C (single bond) ────────────────────────────────────────────────
    if ep == frozenset(["C", "C"]) and bo_ij == 1:
        return "C-C (cyclopropane)" if (is_in_3ring(i) and is_in_3ring(j)) else "C-C (alkane)"

    # ── Fallback: keep original atom-bond-name behaviour ─────────────────
    bo_sym = BOND_ORDER_SYMBOLS.get(bo_ij, "-")
    return "%s%s%s" % (si, bo_sym, sj)


# Predefined colors for functional group contexts
FUNCTIONAL_GROUP_COLORS = {
    # ── O-H (reds) ────────────────────────────────────────────────────────
    "O-H (alcohol)": "#E63946",
    "O-H (phenol)": "#C1121F",
    "O-H (carboxylic acid)": "#FF6B6B",
    "O-H (enol)": "#FF8FA3",
    "O-H (hydroperoxide)": "#9D0208",
    "O-H (hydroxamic acid / oxime)": "#F48C8C",
    "O-H (phosphoric acid)": "#E76F51",
    "O-H (sulfonic acid)": "#F4A261",
    "O-H (boronic acid)": "#FFBA8A",
    # ── N-H (oranges) ─────────────────────────────────────────────────────
    "N-H (primary amine)": "#F4A261",
    "N-H (secondary amine)": "#E76F51",
    "N-H (amide)": "#F9844A",
    "N-H (carbamate/urethane)": "#FB8B24",
    "N-H (urea)": "#F9C784",
    "N-H (sulfonamide)": "#FCBA04",
    "N-H (hydrazine/hydrazone)": "#F3722C",
    "N-H (guanidine/amidine)": "#F8961E",
    "N-H (pyrrole/indole)": "#F9C74F",
    "N-H (aniline, primary)": "#FFD166",
    # ── C-H (greens) ──────────────────────────────────────────────────────
    "C-H (CH3, methyl)": "#2A9D8F",
    "C-H (CH2, methylene)": "#52B788",
    "C-H (CH, methine)": "#74C69D",
    "C-H (sp3)": "#95D5B2",
    "C-H (aromatic)": "#1B4332",
    "C-H (heteroaromatic)": "#2D6A4F",
    "C-H (vinyl/sp2)": "#40916C",
    "C-H (aldehyde)": "#B7E4C7",
    "C-H (sp, alkynyl)": "#D8F3DC",
    "C-H (CH2, cyclopropane)": "#52B788",
    "C-H (CH3, alpha-carbonyl)": "#2A9D8F",
    "C-H (CH2, alpha-carbonyl)": "#52B788",
    "C-H (CH1, alpha-carbonyl)": "#74C69D",
    "C-H (CH3, alpha-ether/alcohol)": "#2A9D8F",
    "C-H (CH2, alpha-ether/alcohol)": "#52B788",
    "C-H (CH1, alpha-ether/alcohol)": "#74C69D",
    "C-H (CH3, alpha-amine)": "#2A9D8F",
    "C-H (CH2, alpha-amine)": "#52B788",
    "C-H (CH1, alpha-amine)": "#74C69D",
    "C-H (CH3, alpha-thioether)": "#2A9D8F",
    "C-H (CH2, alpha-thioether)": "#52B788",
    "C-H (CH3, alpha-halo)": "#95D5B2",
    "C-H (CH2, alpha-halo)": "#B7E4C7",
    "C-H (CH3, alpha-thiocarbonyl)": "#40916C",
    "C-H (CH2, alpha-thiocarbonyl)": "#52B788",
    # ── S-H / P-H / Si-H / B-H (yellows) ─────────────────────────────────
    "S-H (thiol)": "#E9C46A",
    "S-H (thiophenol)": "#F4D35E",
    "P-H (phosphine)": "#A8DADC",
    "Si-H (silane)": "#CDB4DB",
    "B-H (borane)": "#FFC8DD",
    # ── C=O (blues) ───────────────────────────────────────────────────────
    "C=O (ketone)": "#264653",
    "C=O (aldehyde)": "#2B6CB0",
    "C=O (carboxylic acid)": "#1A3C5E",
    "C=O (ester)": "#3B82F6",
    "C=O (lactone)": "#60A5FA",
    "C=O (amide)": "#5B8DB8",
    "C=O (lactam)": "#7EB5D4",
    "C=O (urea)": "#93C5FD",
    "C=O (carbamate/urethane)": "#BFDBFE",
    "C=O (anhydride)": "#1E3A5F",
    "C=O (carbonate)": "#2563EB",
    "C=O (thioester)": "#1D4ED8",
    "C=O (acyl halide)": "#1E40AF",
    # ── C=C / C=C aromatics / C triple bonds (purples) ────────────────────
    "C=C (alkene)": "#7B2D8B",
    "C=C (aromatic)": "#4C1D95",
    "C=C (heteroaromatic)": "#6D28D9",
    "C=C (enamine)": "#8B5CF6",
    "C=C (enol/vinyl ether)": "#A78BFA",
    "C=C (allene/cumulene)": "#C4B5FD",
    "C=C (alkyne)": "#5B21B6",
    "C=N (nitrile)": "#7C3AED",
    # ── C=N / C=S / N= / N triple (violet / magenta) ─────────────────────
    "C=S (thioamide)": "#9D4EDD",
    "C=S (thiocarbonyl/thioketone)": "#C77DFF",
    "C=N (imine/Schiff base)": "#9B72CF",
    "C=N (oxime)": "#C77DFF",
    "C=N (hydrazone)": "#E0AAFF",
    "C=N (imine, ring)": "#9D4EDD",
    "N=N (azo)": "#FF006E",
    "N=N (diazonium)": "#FF073A",
    "N-N (hydrazine)": "#FF4081",
    "N=O (nitroso)": "#FB5607",
    "N=O (nitro group)": "#FF4D6D",
    "N-O (N-oxide)": "#FF7F51",
    "N-O (hydroxylamine)": "#FF9F1C",
    # ── Isocyanate / isothiocyanate / carbodiimide ────────────────────────
    "N=C=O (isocyanate)": "#FFBF69",
    "N=C=S (isothiocyanate)": "#FFCA3A",
    "N=C=N (carbodiimide)": "#FFD166",
    # ── O-O / peroxides ───────────────────────────────────────────────────
    "O-O (peroxide)": "#D62828",
    "O-O (hydroperoxide)": "#E84855",
    # ── S-S / C=S / S=O ──────────────────────────────────────────────────
    "S-S (disulfide)": "#9D7919",
    "S=O (sulfoxide)": "#FFB703",
    "S=O (sulfone)": "#FB8500",
    "S=O (sulfonic acid)": "#F48C06",
    "S=O (sulfonamide)": "#E85D04",
    # ── C-S ───────────────────────────────────────────────────────────────
    "C-S (thiol)": "#CCA43B",
    "C-S (thioether)": "#B08A29",
    "C-S (disulfide)": "#8B6914",
    "C-S (sulfoxide/sulfone)": "#5C4A1E",
    "C-S (thiocyanate)": "#A07430",
    # ── C-O (salmon/peach) ────────────────────────────────────────────────
    "C-O (alcohol)": "#E76F51",
    "C-O (phenol)": "#C4553D",
    "C-O (ether)": "#F4A261",
    "C-O (epoxide)": "#FF7043",
    "C-O (ester)": "#F9844A",
    "C-O (vinyl ether/enol ether)": "#FFAB76",
    "C-O (peroxide)": "#D62828",
    # ── C-N (olive/lime) ──────────────────────────────────────────────────
    "C-N (primary amine)": "#A7C957",
    "C-N (secondary amine)": "#6A994E",
    "C-N (tertiary amine)": "#386641",
    "C-N (aniline/aromatic amine)": "#4F772D",
    "C-N (amide)": "#90A955",
    "C-N (carbamate)": "#B5C18E",
    "C-N (sulfonamide)": "#606C38",
    # ── C-C ───────────────────────────────────────────────────────────────
    "C-C (alkane)": "#AAAAAA",
    "C-C (cyclopropane)": "#888888",
    # ── Halogens (teals / cyans) ──────────────────────────────────────────
    "C-F (alkyl halide)": "#48CAE4",
    "C-F (aryl halide)": "#00B4D8",
    "C-F (vinyl halide)": "#90E0EF",
    "C-F (acyl halide)": "#CAF0F8",
    "C-Cl (alkyl halide)": "#0096C7",
    "C-Cl (aryl halide)": "#0077B6",
    "C-Cl (vinyl halide)": "#00B4D8",
    "C-Cl (acyl halide)": "#023E8A",
    "C-Br (alkyl halide)": "#415A77",
    "C-Br (aryl halide)": "#1B263B",
    "C-Br (vinyl halide)": "#778DA9",
    "C-Br (acyl halide)": "#0D1B2A",
    "C-I (alkyl halide)": "#6B4226",
    "C-I (aryl halide)": "#4A1E0D",
    # ── Si / B / P ────────────────────────────────────────────────────────
    "Si-O (silyl ether/siloxane)": "#CDB4DB",
    "Si-C (organosilane)": "#B5A0C4",
    "B-O (boronic acid)": "#FFC8DD",
    "B-O (boronate ester)": "#FFB3C6",
    "P=O (phosphate/phosphonate)": "#90E0EF",
    "P=O (phosphonic acid)": "#ADE8F4",
    "P=O (phosphonamide)": "#CAF0F8",
    "P-O (phosphate ester)": "#48CAE4",
    # ── fallback ──────────────────────────────────────────────────────────
    "mixed": "#888888",
    "unclassified": "#DDDDDD",
}

_FG_PALETTE = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#264653",
    "#A8DADC",
    "#6D6875",
    "#B5838D",
    "#E76F51",
    "#F77F00",
    "#90BE6D",
    "#43AA8B",
    "#4D908E",
    "#577590",
]


def save_ir_plot_functional_groups(
    results: Dict,
    output_file: str = "ir_spectrum_functional_groups.png",
    freq_range: tuple = (400, 4000),
    broadening: float = 10.0,
    dpi: int = 300,
    top_n_labels: int = 10,
):
    """Save an IR spectrum plot color-coded by functional group context.

    Each vibrational peak is assigned a chemical functional-group label
    (e.g. "CH₃ (α-carbonyl)", "C=O (ketone)", "O-H (alcohol)") by examining
    the bonding environment of the dominant atoms.  Modes are color-coded by
    functional group in both a stick spectrum and a Lorentzian-broadened
    spectrum.

    Args:
        results: Dictionary from ``compute_ir_spectrum()`` that contains a
            ``mode_classification`` key (produced by
            ``classify_vibrational_modes``).
        output_file: Output image filename (png, pdf, svg, etc.)
        freq_range: (min_freq, max_freq) window in cm⁻¹.
        broadening: Lorentzian half-width at half-maximum in cm⁻¹.
        dpi: Image resolution.
        top_n_labels: Number of top-intensity peaks to annotate with labels.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_info("Warning: matplotlib not available. Skipping plot generation.")
        log_info("Install with: pip install matplotlib")
        return

    classification = results.get("mode_classification")
    if not classification or not classification.get("modes"):
        log_info(
            "Warning: no mode_classification found in results. "
            "Run classify_vibrational_modes() first."
        )
        return

    topology = classification.get("topology")
    if topology is None:
        log_info("Warning: topology not found in mode_classification.")
        return

    symbols = results["symbols"]
    frequencies = np.asarray(results["frequencies"])
    intensities = np.asarray(results["intensities_normalized"])

    vib_mask = np.abs(frequencies) > 50.0
    vib_indices = np.where(vib_mask)[0]
    frequencies = frequencies[vib_mask]
    intensities = intensities[vib_mask]

    # Build mode index -> classification lookup and compute fg context
    from .functional_groups import get_functional_group_for_atoms, identify_functional_groups

    fg_map = identify_functional_groups(symbols, topology)

    mode_lookup = {}
    for m in classification["modes"]:
        fg = None
        if fg_map:
            atom_idxs = [
                int(a) if not isinstance(a, (list, tuple)) else int(a[0])
                for a in m["dominant_atoms"]
            ]
            fg = get_functional_group_for_atoms(fg_map, atom_idxs, symbols, topology)
        if not fg:
            fg = _get_functional_group_context(m["dominant_atoms"], symbols, topology, m["type"])
        mode_lookup[m["mode_index"]] = {**m, "_fg": fg}

    # Assign colors (predefined where available, dynamic otherwise)
    dynamic_colors: Dict[str, str] = {}
    palette_idx = [0]

    def get_fg_color(fg: str) -> str:
        if fg in FUNCTIONAL_GROUP_COLORS:
            return FUNCTIONAL_GROUP_COLORS[fg]
        if fg not in dynamic_colors:
            dynamic_colors[fg] = _FG_PALETTE[palette_idx[0] % len(_FG_PALETTE)]
            palette_idx[0] += 1
        return dynamic_colors[fg]

    # Collect functional groups present in the freq range
    in_range_modes = [
        mode_lookup[m["mode_index"]]
        for m in classification["modes"]
        if freq_range[0] <= m["frequency"] <= freq_range[1] and m["mode_index"] in mode_lookup
    ]

    # Collect unique FGs ordered by first occurrence (ascending frequency)
    seen_fg: Dict[str, float] = {}
    for m in sorted(in_range_modes, key=lambda x: x["frequency"]):
        fg = m["_fg"]
        if fg not in seen_fg:
            seen_fg[fg] = m["frequency"]
            get_fg_color(fg)  # register color in consistent order

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ── Stick spectrum ──────────────────────────────────────────────────────
    for freq, intensity, orig_idx in zip(frequencies, intensities, vib_indices):
        if not (freq_range[0] <= freq <= freq_range[1]):
            continue
        m_info = mode_lookup.get(int(orig_idx))
        fg = m_info["_fg"] if m_info else "unclassified"
        color = get_fg_color(fg)
        is_inter = m_info.get("is_intermolecular", False) if m_info else False
        ls = "--" if is_inter else "-"
        ax1.vlines(freq, 0, intensity, colors=color, linewidth=1.8, alpha=0.85, linestyles=ls)
        marker = "D" if is_inter else "o"
        ax1.plot(freq, intensity, marker, color=color, markersize=3)

    # Annotate top-N most intense peaks
    top_modes = sorted(in_range_modes, key=lambda x: x["intensity"], reverse=True)[:top_n_labels]
    for ann_idx, m in enumerate(top_modes):
        fg = m["_fg"]
        color = get_fg_color(fg)
        y_offset = 0.05 + 0.035 * (ann_idx % 4)
        ax1.annotate(
            _peak_label(m),
            xy=(m["frequency"], m["intensity"]),
            xytext=(m["frequency"], m["intensity"] + y_offset),
            fontsize=7,
            ha="center",
            va="bottom",
            rotation=45,
            color=color,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.set_ylabel("Normalized Intensity", fontsize=12)
    ax1.set_title("IR Stick Spectrum — Functional Group Assignment", fontsize=13, fontweight="bold")
    ax1.set_xlim(freq_range)
    ax1.set_ylim(0, 1.45)
    ax1.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=get_fg_color(fg), lw=2.5, label=fg)
        for fg in sorted(seen_fg, key=lambda g: seen_fg[g])
    ]
    if legend_elements:
        ncols = max(1, len(legend_elements) // 8)
        ax1.legend(
            handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.85, ncol=ncols
        )

    # ── Lorentzian-broadened spectrum ───────────────────────────────────────
    freq_axis = np.linspace(freq_range[0], freq_range[1], int(freq_range[1] - freq_range[0]))
    spectrum = np.zeros_like(freq_axis)
    for freq, intensity in zip(frequencies, intensities):
        if freq_range[0] <= freq <= freq_range[1]:
            spectrum += intensity * broadening**2 / ((freq_axis - freq) ** 2 + broadening**2)

    ax2.plot(freq_axis, spectrum, color="#333333", linewidth=1.5)
    ax2.fill_between(freq_axis, spectrum, alpha=0.12, color="#333333")

    # Colored triangle markers at peaks with intensity > 5 %
    intensity_threshold = 0.05
    for freq, intensity, orig_idx in zip(frequencies, intensities, vib_indices):
        if not (freq_range[0] <= freq <= freq_range[1]):
            continue
        if intensity < intensity_threshold:
            continue
        m_info = mode_lookup.get(int(orig_idx))
        fg = m_info["_fg"] if m_info else "unclassified"
        color = get_fg_color(fg)
        idx_near = np.argmin(np.abs(freq_axis - freq))
        ax2.plot(freq, spectrum[idx_near], "v", color=color, markersize=6, alpha=0.9, zorder=5)

    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Intensity (arbitrary units)", fontsize=12)
    ax2.set_title(
        f"IR Spectrum (Lorentzian, \u03b3={broadening:.1f} cm\u207b\u00b9) "
        "— Functional Group Peaks",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_xlim(freq_range)
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=get_fg_color(fg), lw=2.5, label=fg)
        for fg in sorted(seen_fg, key=lambda g: seen_fg[g])
    ]
    if legend_elements:
        ncols = max(1, min(len(legend_elements), 5))
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncols,
            fontsize=8,
            framealpha=0.85,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12 + 0.035 * ((len(legend_elements) - 1) // ncols))
    else:
        plt.tight_layout()

    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    log_info(f"Functional group IR plot saved to {output_file}")


def compute_ir_all_conformers(
    xyz_file: str,
    potential_name: str = "so3lr",
    model_path: Optional[str] = None,
    temperature: float = 300.0,
    charge: float = 0.0,
    lr_cutoff: float = 1000.0,
    use_float64: bool = False,
    output_prefix: str = "conformer",
    optimize: bool = True,
    fmax: float = 0.01,
    fd_hessian: bool = False,
    fd_displacement: float = 0.01,
    potential_options: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
    classify: bool = True,
    save_mode_xyz: bool = False,
    save_hessian: bool = False,
    save_structure: bool = False,
    save_analysis: bool = True,
) -> list:
    """Compute IR spectra for all conformers in a multi-frame XYZ file.

    Per conformer ``i`` the following files are written, all sharing the
    ``{output_prefix}_{i}_`` stem:

    - ``ir_spectrum.dat`` / ``ir_spectrum.png`` — always.
    - ``normal_modes.xyz``       — when ``save_mode_xyz``.
    - ``ir_hessian_data.txt``    — when ``save_hessian``.
    - ``optimized_structure.xyz``— when ``save_structure``.
    - ``mode_analysis.txt``      — when ``save_analysis`` (and modes were
      classified).

    Args:
        xyz_file: Path to multi-frame XYZ file
        potential_name: Registered potential name (default: ``"so3lr"``).
        model_path: Optional path to custom model
        temperature: Temperature in Kelvin
        charge: Total molecular charge
        lr_cutoff: Long-range cutoff in Angstrom
        use_float64: Use float64 precision
        output_prefix: Prefix for output files
        optimize: If True, optimize each geometry before computing Hessian
        fmax: Force convergence criterion for optimization
        fd_hessian: If True, approximate Hessian by finite differences
        fd_displacement: Displacement step in Angstrom for finite-differences Hessian
        atom_indices: Optional 0-based subset of atoms for a partial
            Hessian (same semantics as :func:`compute_ir_spectrum`).
        classify: If True, classify vibrational modes (needed for the
            per-mode analysis output).
        save_mode_xyz: If True, write a per-conformer normal-modes XYZ.
        save_hessian: If True, write per-conformer Hessian data.
        save_structure: If True, write the per-conformer optimized geometry.
        save_analysis: If True (default), write the per-conformer mode
            analysis when classification is available.

    Returns:
        List of results dictionaries, one per conformer
    """
    structures = load_multiframe_xyz(xyz_file)
    n_conformers = len(structures)

    log_header("IR SPECTRA FOR MULTIPLE CONFORMERS")
    log_info(f"Found {n_conformers} conformers")
    log_info("")

    all_results = []

    for i in tqdm(range(n_conformers), desc="Processing conformers", unit="conformer"):
        tqdm.write(f"{'='*70}")
        tqdm.write(f"Processing conformer {i+1}/{n_conformers}")
        tqdm.write(f"{'='*70}")

        # Compute IR for this conformer
        results = compute_ir_spectrum(
            xyz_file,
            potential_name=potential_name,
            model_path=model_path,
            temperature=temperature,
            charge=charge,
            lr_cutoff=lr_cutoff,
            use_float64=use_float64,
            conformer_index=i,
            optimize=optimize,
            fmax=fmax,
            classify=classify,
            fd_hessian=fd_hessian,
            fd_displacement=fd_displacement,
            potential_options=potential_options,
            atom_indices=atom_indices,
        )

        # Save individual outputs (always: spectrum data + plot)
        save_ir_spectrum(results, f"{output_prefix}_{i}_ir_spectrum.dat")
        save_ir_plot(results, f"{output_prefix}_{i}_ir_spectrum.png")

        # Optional per-conformer Hessian-based outputs, mirroring the
        # single-conformer CLI path so --mode-xyz / --save-hessian /
        # --save-structure are honored with --all-conformers too.
        if save_mode_xyz:
            save_normal_mode_xyz(results, output_file=f"{output_prefix}_{i}_normal_modes.xyz")
        if save_hessian:
            save_ir_hessian_data(results, f"{output_prefix}_{i}_ir_hessian_data.txt")
        if save_structure:
            save_optimized_structure(results, f"{output_prefix}_{i}_optimized_structure.xyz")
        if save_analysis and results.get("mode_classification") is not None:
            save_mode_analysis(results, f"{output_prefix}_{i}_mode_analysis.txt")

        all_results.append(results)
        tqdm.write("")

    log_info(f"{'='*70}")
    log_info(f"Completed IR calculations for {n_conformers} conformers")
    log_info(f"{'='*70}")

    return all_results
