"""
ML potentials provided by upstream jax-md.

Wraps the pure-JAX ports that live in ``jax_md._nn`` (jax-md main, Aug 2026)
as MARS :class:`~mars.potentials.PotentialWrapper` objects:

* ``uma``     — FAIR-Chem UMA (``jax_md._nn.uma``).
* ``orb``     — Orb-v3 conservative OMol (``jax_md._nn.orb``).
* ``ani``     — ANI-2x (``jax_md._nn.ani``).
* ``aimnet2`` — AIMNet2 (``jax_md._nn.aimnet2``).
* ``aceff``   — AceFF 1.1 / 2.0 (``jax_md._nn.aceff``).
* ``so3lr_jaxmd`` — jax-md's own SO3LR port (``jax_md._nn.so3lr``), an
  alternative to the ``so3lr``-package backend of :class:`SO3LRPotential`.

All of them run in free space (``space.free()``) with all-pairs neighbor
lists, like the other MARS potentials.  Two conventions keep them drop-in
compatible with the sampling / optimisation / IR loops:

* single-cutoff models use one jax-md ``NeighborList`` as neighbor state;
* two-cutoff models (AIMNet2, SO3LR) use a ``(short, long_range)`` tuple, the
  same layout as the ``so3lr`` potential, so the existing overflow checks and
  replica padding work unchanged.

``get_neighbor_kwargs`` hands plain index arrays to the energy function; the
energy function rebuilds the jax-md neighbor object from a template (the
models only read ``idx`` and ``format``).

Upstream maturity (audited Oct 2026) is recorded on each class as
``UPSTREAM`` and printed once through :func:`_warn_experimental` until the
model passes the MARS validation suite (``validation/`` at the repo root).
"""

import warnings
from dataclasses import replace
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .potentials import OMOL25_ELEMENTS, PotentialWrapper, register_potential, validate_model_inputs

# Effective box for free space.  With ``disable_cell_list=True`` the box is
# never used to wrap coordinates, but the jax-md builders insist on one.
_FREE_BOX = 1.0e4

_WARNED = set()


def _warn_experimental(name: str, note: str):
    """Emit a one-time warning for a potential not yet validated in MARS."""
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"The '{name}' potential is experimental in MARS: {note} "
        "See validation/README.md for its current validation status.",
        UserWarning,
        stacklevel=3,
    )


def _require(module: str, feature: str, hint: str):
    """Import *module* or raise an ImportError that says how to get it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"The {feature} potential requires '{module}', which ships with "
            f"jax-md main (>= commit a41c7d1, Aug 2026).\n{hint}\n(import error: {exc})"
        )


_JAXMD_MAIN_HINT = (
    "Install with:\n"
    "  pip install 'git+https://github.com/jax-md/jax-md.git'\n"
    "  pip install equinox            # ORB / ANI / AIMNet2 / AceFF / SO3LR ports"
)


def _weights_hint(family: str) -> str:
    return (
        f"Fetch the '{family}' checkpoint with:\n"
        f"  python -m jax_md._nn.weights {family}\n"
        "If that fails with HTTP 404 (the upstream 'weights-v1' release is not\n"
        "published yet), download the Git LFS files instead; see\n"
        "docs/install.md, section 'jax-md model weights'."
    )


class _JaxMDModelPotential(PotentialWrapper):
    """Common free-space plumbing for the jax-md neighbor-list models.

    Subclasses implement :meth:`_make_fns` and return
    ``(neighbor_fn, model_energy_fn, paired)`` where ``model_energy_fn`` has
    the jax-md signature ``(positions, neighbor) -> eV`` and ``paired`` says
    whether ``neighbor_fn`` allocates a ``PairedNeighborList``.
    """

    #: Registry name (used in warnings).
    NAME = "jaxmd"
    #: One-line upstream maturity note; ``None`` once validated in MARS.
    UPSTREAM = None
    #: Whether the model takes a total charge / a spin multiplicity. Models
    #: without an input refuse non-default values instead of ignoring them.
    CHARGE_AWARE = True
    SPIN_AWARE = False
    #: Atomic numbers the model was trained on (``None``: no limit).
    ELEMENTS = None
    #: Name used in error messages (defaults to ``NAME``).
    MODEL_LABEL = None

    def supported_elements(self):
        """Atomic numbers this model supports (``None`` for no restriction)."""
        return self.ELEMENTS

    #: How to read the model's partial charges (for IR intensities), or None.
    CHARGE_TAP: "_ChargeTap | None" = None

    def __init__(
        self,
        species,
        charge: float = 0.0,
        spin: float = 1.0,
        dtype=None,
        dr_threshold: float = 0.5,
        capacity_multiplier: float = 1.25,
        **kwargs,
    ):
        if dtype in (jnp.float64, "float64"):
            jax.config.update("jax_enable_x64", True)
        self._species = np.asarray(species, dtype=int)
        self._n_atoms = len(self._species)
        self._charge = float(charge)
        self._spin = float(spin)  # multiplicity 2S+1
        validate_model_inputs(
            self.MODEL_LABEL or self.NAME,
            self._species,
            self._charge,
            self._spin,
            elements=self.supported_elements(),
            charge_aware=self.CHARGE_AWARE,
            spin_aware=self.SPIN_AWARE,
        )
        self._dr_threshold = float(dr_threshold)
        self._capacity_multiplier = float(capacity_multiplier)
        self._initialized = False
        self._paired = False
        if self.UPSTREAM:
            _warn_experimental(self.NAME, self.UPSTREAM)

    # -- subclass hook --------------------------------------------------
    def _make_fns(self, displacement_fn, box):
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------
    def initialize(self, positions: jnp.ndarray):
        from jax_md import space

        positions = jnp.asarray(positions)
        displacement_fn, _ = space.free()
        box = jnp.eye(3) * _FREE_BOX
        self._neighbor_fn, self._model_energy_fn, self._paired = self._make_fns(
            displacement_fn, box
        )
        nbrs = self._neighbor_fn.allocate(positions)
        if self._paired:
            self._template = (nbrs.short, nbrs.long_range)
        else:
            self._template = nbrs
        self._initialized = True

    def _build_energy_fn(self) -> Callable:
        if not self._initialized:
            raise RuntimeError(
                f"{type(self).__name__} not initialized. Call initialize(positions) first."
            )
        model_energy_fn = self._model_energy_fn
        template = self._template

        if self._paired:
            from jax_md.partition import PairedNeighborList

            short_t, long_t = template

            def energy_fn(positions, neighbor=None, neighbor_lr=None, **kwargs):
                short = short_t if neighbor is None else replace(short_t, idx=neighbor)
                long_ = long_t if neighbor_lr is None else replace(long_t, idx=neighbor_lr)
                return model_energy_fn(positions, PairedNeighborList(short, long_))

        else:

            def energy_fn(positions, neighbor=None, **kwargs):
                nbrs = template if neighbor is None else replace(template, idx=neighbor)
                return model_energy_fn(positions, nbrs)

        return energy_fn

    def build_energy_fn(self, with_charges: bool = False):
        energy_fn = self._build_energy_fn()
        if not with_charges:
            return energy_fn
        if self.CHARGE_TAP is None:
            return super().build_energy_fn(with_charges=True)
        return energy_fn, self.CHARGE_TAP(energy_fn)

    # -- neighbor interface ---------------------------------------------
    @property
    def uses_neighbor_lists(self) -> bool:
        return True

    def allocate_neighbors(self, positions: jnp.ndarray, extra_capacity: int = 0):
        positions = jnp.asarray(positions)
        nbrs = self._neighbor_fn.allocate(positions, extra_capacity=extra_capacity)
        if self._paired:
            return (nbrs.short, nbrs.long_range)
        return nbrs

    def update_neighbors(self, positions: jnp.ndarray, nbr_state):
        positions = jnp.asarray(positions)
        if self._paired:
            short, long_ = nbr_state
            return (short.update(positions), long_.update(positions))
        return nbr_state.update(positions)

    def get_neighbor_kwargs(self, nbr_state) -> Dict[str, Any]:
        if self._paired:
            return {"neighbor": nbr_state[0].idx, "neighbor_lr": nbr_state[1].idx}
        return {"neighbor": nbr_state.idx}

    def _nl_kwargs(self) -> Dict[str, Any]:
        return {
            "dr_threshold": self._dr_threshold,
            "capacity_multiplier": self._capacity_multiplier,
            "disable_cell_list": True,
        }


def _load_checkpoint_or_explain(loader, family: str):
    """Run *loader* and turn a missing checkpoint into an actionable error."""
    try:
        return loader()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}\n{_weights_hint(family)}") from exc


class _ChargeTap:
    """Record the partial charges a jax-md model computes inside its energy call.

    The jax-md ports compute charges internally (for their Coulomb terms) but
    only return the energy.  This temporarily wraps the method that produces
    them, so a single energy evaluation also yields the charges, under ``jit``
    / ``grad`` too (the wrapper is only active while the function is traced).

    Args:
        module: Module path of the model, e.g. ``'jax_md._nn.aimnet2.model'``.
        cls: Class name owning the method.
        method: Method name to wrap.
        pick: ``(out, owner_instance) -> charges`` applied to the method's return
            value of the last call.
    """

    def __init__(self, module, cls, method, pick=lambda out, this: out):
        self.module, self.cls, self.method, self.pick = module, cls, method, pick

    def __call__(self, energy_fn):
        import importlib

        owner = getattr(importlib.import_module(self.module), self.cls)
        method, pick = self.method, self.pick

        def charges_fn(positions, **kwargs):
            seen = []
            original = getattr(owner, method)

            def recording(this, *args, **kw):
                out = original(this, *args, **kw)
                seen.append((out, this))
                return out

            setattr(owner, method, recording)
            try:
                energy_fn(positions, **kwargs)
            finally:
                setattr(owner, method, original)
            if not seen:
                raise RuntimeError(f"{self.cls}.{method} was not called; cannot read charges.")
            out, this = seen[-1]
            return jnp.reshape(pick(out, this), (-1,))

        return charges_fn


# ============================================================================
# ORB / ANI / AIMNet2 / AceFF / SO3LR (bio-mlff port, jax-md PR #411)
# ============================================================================


@register_potential("orb")
class ORBPotential(_JaxMDModelPotential):
    """Orb-v3 conservative OMol potential (jax-md port of ``orb-models``).

    Charge- and spin-aware (``spin`` is the multiplicity, as in orb-models).

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model: jax-md checkpoint name (default ``'orb-jax-v3-conservative-omol'``).
        model_path: Optional local ``.eqx`` checkpoint overriding ``model``.
        charge: Total charge.
        spin: Spin multiplicity (1 = singlet).
    """

    NAME = "orb"
    MODEL_LABEL = "Orb-v3 OMol"
    ELEMENTS = OMOL25_ELEMENTS
    SPIN_AWARE = True  # multiplicity, as orb-models' ``atoms.info["spin"]``
    UPSTREAM = (
        "upstream jax-md checks it against orb-models on one molecule only " "(PR #411, @mitkotak)."
    )

    def __init__(self, species, model="orb-jax-v3-conservative-omol", model_path=None, **kw):
        super().__init__(species, **kw)
        self._model = model
        self._model_path = model_path

    def _make_fns(self, displacement_fn, box):
        energy = _require("jax_md.energy", "orb", _JAXMD_MAIN_HINT)
        _require("jax_md._nn.orb", "orb", _JAXMD_MAIN_HINT)
        neighbor_fn, efn = _load_checkpoint_or_explain(
            lambda: energy.orb_neighbor_list(
                displacement_fn,
                box,
                species=jnp.asarray(self._species),
                model=self._model,
                model_path=self._model_path,
                total_charge=self._charge,
                total_spin=self._spin,
                **self._nl_kwargs(),
            ),
            "orb",
        )
        return neighbor_fn, efn, False


@register_potential("ani")
class ANIPotential(_JaxMDModelPotential):
    """ANI-2x potential (jax-md port of TorchANI).

    Neutral, closed-shell molecules of H, C, N, O, F, S and Cl only.  The
    ``charge`` / ``spin`` arguments are accepted for interface uniformity and
    must be 0 / 1.

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model: ``'ani2x-jax-ensemble'`` (8-model mean, default) or a single
            member checkpoint name.
        model_path: Optional local ``.eqx`` checkpoint.
    """

    NAME = "ani"
    MODEL_LABEL = "ANI-2x"
    ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})
    CHARGE_AWARE = False
    UPSTREAM = (
        "upstream jax-md checks it against TorchANI on one molecule only " "(PR #411, @mitkotak)."
    )

    def __init__(self, species, model="ani2x-jax-ensemble", model_path=None, **kw):
        super().__init__(species, **kw)
        self._model = model
        self._model_path = model_path

    def _make_fns(self, displacement_fn, box):
        energy = _require("jax_md.energy", "ani", _JAXMD_MAIN_HINT)
        _require("jax_md._nn.ani", "ani", _JAXMD_MAIN_HINT)
        neighbor_fn, efn = _load_checkpoint_or_explain(
            lambda: energy.ani_neighbor_list(
                displacement_fn,
                box,
                species=jnp.asarray(self._species),
                model=self._model,
                model_path=self._model_path,
                **self._nl_kwargs(),
            ),
            "ani",
        )
        # jax-md converts Hartree to eV with CODATA 2018 (27.211386245988);
        # TorchANI (and ANI-2x as published) uses 27.211386024367243. Rescale
        # so MARS reproduces TorchANI exactly (a relative 8e-9 shift).
        from jax_md.units import HARTREE_EV

        scale = 27.211386024367243 / float(HARTREE_EV)

        def ani_energy(position, neighbor, **kwargs):
            return scale * efn(position, neighbor, **kwargs)

        return neighbor_fn, ani_energy, False


def _stable_d3_pair_energy(
    pair_c6ab,
    nci,
    ncj,
    rij,
    r2r4_i,
    r2r4_j,
    *,
    d3_s6,
    d3_s8,
    d3_a1,
    d3_a2,
    d3_k3,
):
    """DFT-D3(BJ) pair energy with the C6 interpolation of the original ``aimnet``.

    jax-md's ``d3_pair_energy`` exponentiates ``k3 * dCN^2`` directly. For
    atoms whose coordination number is far from every D3 reference (hypervalent
    S and P, e.g. sulfonates and phosphates) all weights underflow, C6 falls to
    0, the dispersion term disappears (energy errors of 0.1-0.2 eV) and the
    square root of the resulting 0 gives NaN forces. ``aimnet`` subtracts the
    largest exponent over the valid references first and drops references more
    than ``exp(-12)`` below it; this reproduces that exactly
    (``aimnet.modules.lr.DFTD3._calc_torch_c6ij``).
    """
    reference_c6 = pair_c6ab[..., 0]
    reference_cn_i = pair_c6ab[..., 1]
    reference_cn_j = pair_c6ab[..., 2]
    n_ref = reference_c6.shape[-2] * reference_c6.shape[-1]

    exp_arg = d3_k3 * (
        (reference_cn_i - nci[..., None, None]) ** 2 + (reference_cn_j - ncj[..., None, None]) ** 2
    )
    exp_arg = exp_arg.reshape(*rij.shape, n_ref)
    reference_c6 = reference_c6.reshape(*rij.shape, n_ref)
    valid = reference_c6 != 0.0
    max_exp = jnp.max(jnp.where(valid, exp_arg, -jnp.inf), axis=-1, keepdims=True)
    finite_max = jnp.isfinite(max_exp)
    shifted = jnp.where(finite_max & valid, exp_arg - jnp.where(finite_max, max_exp, 0.0), 0.0)
    keep = valid & finite_max & (shifted >= -12.0)
    weights = jnp.where(keep, jnp.exp(jnp.where(keep, shifted, 0.0)), 0.0)
    weight_sum = jnp.sum(weights, axis=-1)
    c6_sum = jnp.sum(weights * reference_c6, axis=-1)
    has_c6 = weight_sum > 1.0e-12
    c6 = jnp.where(has_c6, c6_sum / jnp.where(has_c6, weight_sum, 1.0), 0.0)
    c8 = 3.0 * c6 * r2r4_i * r2r4_j

    ratio = jnp.where(c6 > 0.0, c8 / jnp.where(c6 > 0.0, c6, 1.0), 0.0)
    bj_radius = d3_a1 * jnp.sqrt(jnp.where(ratio > 0.0, ratio, 1.0)) * (ratio > 0.0) + d3_a2
    bj_radius2 = bj_radius**2
    bj_radius6 = bj_radius2**3
    bj_radius8 = bj_radius6 * bj_radius2
    e6 = -d3_s6 * c6 / (rij**6 + bj_radius6)
    e8 = -d3_s8 * c8 / (rij**8 + bj_radius8)
    return e6 + e8


def _patch_aimnet2_d3():
    """Install :func:`_stable_d3_pair_energy` in jax-md's AIMNet2 module (idempotent)."""
    from jax_md._nn.aimnet2 import model as aimnet2_model

    if getattr(aimnet2_model.d3_pair_energy, "__name__", "") != "_stable_d3_pair_energy":
        aimnet2_model.d3_pair_energy = _stable_d3_pair_energy


@register_potential("aimnet2")
class AIMNet2Potential(_JaxMDModelPotential):
    """AIMNet2 potential (jax-md port of ``aimnet``), charge-aware.

    Uses the free-space all-pairs Coulomb branch of the jax-md port
    (``periodic=False``) and a ``(short, long_range)`` neighbor pair.

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model: jax-md checkpoint name (default ``'aimnet2-jax'``).
        model_path: Optional local ``.eqx`` checkpoint.
        charge: Total charge.
    """

    NAME = "aimnet2"
    MODEL_LABEL = "AIMNet2"
    # ``implemented_species`` of the jax-md checkpoint.
    ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})
    # Charges equilibrated by the last interaction layer.
    CHARGE_TAP = _ChargeTap("jax_md._nn.aimnet2.model", "PartialChargesHead", "__call__")
    UPSTREAM = (
        "upstream jax-md checks it against aimnet on one molecule only " "(PR #411, @mitkotak)."
    )

    def __init__(self, species, model="aimnet2-jax", model_path=None, **kw):
        super().__init__(species, **kw)
        self._model = model
        self._model_path = model_path

    def _make_fns(self, displacement_fn, box):
        energy = _require("jax_md.energy", "aimnet2", _JAXMD_MAIN_HINT)
        _require("jax_md._nn.aimnet2", "aimnet2", _JAXMD_MAIN_HINT)
        _patch_aimnet2_d3()
        # The free-space branch builds its own space.free() displacement.
        neighbor_fn, efn = _load_checkpoint_or_explain(
            lambda: energy.aimnet2_neighbor_list(
                None,
                box,
                species=jnp.asarray(self._species),
                model=self._model,
                model_path=self._model_path,
                total_charge=self._charge,
                periodic=False,
                **self._nl_kwargs(),
            ),
            "aimnet2",
        )
        return neighbor_fn, efn, True


def _aceff_effective_charges(history, model):
    q = jnp.concatenate(history, axis=-1)  # (n_atoms, n_layers * n_channels)
    head = getattr(model, "coulomb_head", None)
    if head is None:
        return jnp.mean(q, axis=-1)
    w = head.qweights
    return (q * w[None, :]).sum(axis=-1) / w.sum()


def _aceff_with_coulomb_cutoff(net, coulomb_cutoff):
    """Return *net* with its Coulomb head set to the requested cutoff.

    ``"checkpoint"`` / ``None`` means plain all-pairs Coulomb (the AceFF
    checkpoint default); a number keeps jax-md's reaction-field form with that
    cutoff. AceFF 1.1 has no Coulomb head and is returned unchanged.
    """
    import copy

    import equinox as eqx

    head = getattr(net, "coulomb_head", None)
    if head is None:
        return net
    cutoff = None if coulomb_cutoff in (None, "checkpoint") else float(coulomb_cutoff)
    if head.coulomb_cutoff == cutoff:
        return net
    new_head = copy.copy(head)
    # ``coulomb_cutoff`` is a static (non-array) field, so it is set directly
    # on the copy rather than through ``eqx.tree_at``.
    object.__setattr__(new_head, "coulomb_cutoff", cutoff)
    return eqx.tree_at(lambda n: n.coulomb_head, net, new_head)


@register_potential("aceff")
class AceFFPotential(_JaxMDModelPotential):
    """AceFF potential (jax-md port of the TorchMD-Net TensorNet models).

    AceFF 2.0 has an explicit Coulomb term. The jax-md checkpoint was
    converted with a 12 Å reaction-field cutoff and a solvent dielectric of
    78.3 (the setting bio-mlff used for its references), whereas the AceFF
    checkpoint's own default, and what torchmd-net uses, is plain all-pairs
    Coulomb in vacuum. MARS restores the checkpoint default
    (``coulomb_cutoff="checkpoint"``), which is the right choice for isolated
    molecules and matches torchmd-net.

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model: ``'aceff-jax-2.0'`` (default) or ``'aceff-jax-1.1'``.
        model_path: Optional local ``.eqx`` checkpoint.
        charge: Total charge.
        coulomb_cutoff: ``"checkpoint"`` (default: plain Coulomb, as the AceFF
            checkpoint), ``None`` (same), or a cutoff in Å to use jax-md's
            reaction-field Coulomb (``12.0`` reproduces upstream jax-md).
    """

    NAME = "aceff"
    MODEL_LABEL = "AceFF"
    # Training elements of AceFF 1.1 and 2.0 (Hugging Face model cards).
    ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})
    # AceFF predicts several charge channels per layer; its Coulomb term mixes
    # them with ``qweights``. The same weighted mean is the effective atomic
    # charge used for the dipole (it sums to the total charge).
    CHARGE_TAP = _ChargeTap(
        "jax_md._nn.aceff.model",
        "AceFF",
        "local_node_energies_and_charges",
        pick=lambda out, this: _aceff_effective_charges(out[1], this),
    )
    UPSTREAM = (
        "upstream jax-md checks it against torchmd-net on one molecule only "
        "(PR #411, @mitkotak)."
    )

    def __init__(
        self,
        species,
        model="aceff-jax-2.0",
        model_path=None,
        coulomb_cutoff="checkpoint",
        **kw,
    ):
        super().__init__(species, **kw)
        self._model = model
        self._model_path = model_path
        self._coulomb_cutoff = coulomb_cutoff

    def _make_fns(self, displacement_fn, box):
        _require("jax_md.energy", "aceff", _JAXMD_MAIN_HINT)
        aceff = _require("jax_md._nn.aceff", "aceff", _JAXMD_MAIN_HINT)
        from jax_md import partition

        net = _load_checkpoint_or_explain(
            lambda: aceff.load_model(self._model, model_path=self._model_path), "aceff"
        )
        net = _aceff_with_coulomb_cutoff(net, self._coulomb_cutoff)
        neighbor_fn = partition.neighbor_list(
            displacement_fn, box, float(net.cutoff), **self._nl_kwargs()
        )
        species = jnp.asarray(self._species)
        total_charge = self._charge

        def efn(position, neighbor, **kwargs):
            return net(
                position,
                species,
                displacement_fn=displacement_fn,
                neighbors=neighbor,
                total_charge=total_charge,
            )

        return neighbor_fn, efn, False


@register_potential("so3lr_jaxmd")
class SO3LRJaxMDPotential(_JaxMDModelPotential):
    """SO3LR through jax-md's own port (``jax_md._nn.so3lr``).

    An alternative backend to the ``so3lr`` potential, which uses the
    ``so3lr`` package.  Useful where the ``so3lr`` package cannot be
    installed; numerically the two are validated against each other in
    ``validation/``.

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model_path: Optional local ``.eqx`` checkpoint.
        charge: Total charge.
        spin: Must be 1: the jax-md checkpoint is SO3LR v1, which is not
            spin-aware (use ``so3lr`` with a v2 model for open-shell systems).
    """

    NAME = "so3lr_jaxmd"
    MODEL_LABEL = "SO3LR v1 (jax-md port)"
    ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17})
    # jax-md ships the v1 checkpoint, whose spin input was never trained.
    SPIN_AWARE = False
    CHARGE_TAP = _ChargeTap("jax_md._nn.so3lr.model", "PartialChargesHead", "__call__")
    UPSTREAM = (
        "upstream jax-md checks it against the so3lr package on one molecule only "
        "(PR #411, @mitkotak)."
    )

    def __init__(self, species, model="so3lr", model_path=None, **kw):
        super().__init__(species, **kw)
        self._model = model
        self._model_path = model_path

    def _make_fns(self, displacement_fn, box):
        energy = _require("jax_md.energy", "so3lr_jaxmd", _JAXMD_MAIN_HINT)
        _require("jax_md._nn.so3lr", "so3lr_jaxmd", _JAXMD_MAIN_HINT)
        neighbor_fn, efn = _load_checkpoint_or_explain(
            lambda: energy.so3lr_neighbor_list(
                displacement_fn,
                box,
                species=jnp.asarray(self._species),
                model=self._model,
                model_path=self._model_path,
                total_charge=self._charge,
                total_spin=0.0,
                **self._nl_kwargs(),
            ),
            "so3lr",
        )
        return neighbor_fn, efn, True


# ============================================================================
# UMA (jax-md PRs #376, #394, #400, #401)
# ============================================================================


@register_potential("uma")
class UMAPotential(_JaxMDModelPotential):
    """FAIR-Chem UMA through the pure-JAX port in ``jax_md._nn.uma``.

    Weights are the gated FAIR-Chem checkpoints on Hugging Face
    (``facebook/UMA``): accept the licence on the model page and log in
    (``huggingface-cli login``) once.  The first load converts the PyTorch
    checkpoint (needs ``torch``); later loads use the cached ``.npz``.

    Args:
        species: Atomic numbers, shape ``(n_atoms,)``.
        model: ``'uma-s-1p1'``, ``'uma-s-1p2'`` (default) or ``'uma-m-1p1'``,
            or a path to a checkpoint.
        task: UMA task / energy head: ``'omol'`` (molecules, default),
            ``'omat'``, ``'oc20'``, ``'odac'`` or ``'omc'``.
        charge: Total charge (used by the ``omol`` task).
        spin: Spin multiplicity (used by the ``omol`` task).
        merge_mole: Pre-mix the mixture-of-linear-experts weights for this
            composition (faster; valid because MARS runs one system).
        use_kernels: Use the jax-md Pallas kernels.  ``None`` enables them on
            GPU only.
    """

    NAME = "uma"
    MODEL_LABEL = "UMA"

    def supported_elements(self):
        # omol: the 83 OMol25 elements; the other heads share the 100-element
        # embedding of the checkpoints (``max_num_elements``).
        return OMOL25_ELEMENTS if self._task == "omol" else frozenset(range(1, 101))

    SPIN_AWARE = True  # multiplicity (FAIR-Chem ``atoms.info["spin"]``), omol task only
    UPSTREAM = (
        "upstream jax-md compares UMA energies (not forces) with FAIR-Chem at "
        "atol 0.05 (PR #376, @ekurtulus, @abhijeetgangan)."
    )
    _TASKS = ("omol", "omat", "oc20", "odac", "omc")

    def __init__(
        self,
        species,
        model: str = "uma-s-1p2",
        task: str = "omol",
        merge_mole: bool = True,
        use_kernels=None,
        **kw,
    ):
        task = str(task).lower()
        if task not in self._TASKS:
            raise ValueError(f"Unknown UMA task '{task}'. Choose from: {', '.join(self._TASKS)}")
        self._task = task
        # Only the omol head was trained with charge/spin; the other tasks get
        # charge 0 and spin 0 ("unset"), as in FAIR-Chem.
        self.CHARGE_AWARE = self.SPIN_AWARE = task == "omol"
        self.MODEL_LABEL = f"UMA ({task} task)"
        super().__init__(species, **kw)
        if task != "omol":
            self._spin = 0.0
        self._model = model
        self._merge_mole = bool(merge_mole)
        if use_kernels is None:
            use_kernels = jax.default_backend() == "gpu"
        self._use_kernels = bool(use_kernels)

    def _make_fns(self, displacement_fn, box):
        energy = _require("jax_md.energy", "uma", _JAXMD_MAIN_HINT)
        _require("jax_md._nn.uma", "uma", _JAXMD_MAIN_HINT)
        from jax_md import partition
        from jax_md._nn.uma.featurizer import uma_featurizer

        try:
            neighbor_fn, init_fn, uma_energy = energy.uma_neighbor_list(
                displacement_fn,
                box,
                atoms=jnp.asarray(self._species),
                checkpoint_path=self._model,
                neighbor_list_fn=partition.neighbor_list,
                featurizer_fn=uma_featurizer,
                charge=jnp.array([self._charge]),
                spin=jnp.array([self._spin]),
                head_dataset=self._task,
                use_kernels=self._use_kernels,
                merge_mole=self._merge_mole,
                **self._nl_kwargs(),
            )
        except Exception as exc:
            if "401" in str(exc) or "gated" in str(exc).lower() or "token" in str(exc).lower():
                raise RuntimeError(
                    "UMA weights are gated on Hugging Face. Accept the licence at "
                    "https://huggingface.co/facebook/UMA and run `huggingface-cli login`."
                ) from exc
            raise

        species = jnp.asarray(self._species)
        self._uma_init = init_fn
        self._uma_energy = uma_energy
        self._uma_params = None

        def model_energy_fn(positions, neighbor):
            if self._uma_params is None:
                self._uma_params = init_fn(jax.random.PRNGKey(0), positions, neighbor)
            return uma_energy(self._uma_params, positions, neighbor, atoms=species)

        return neighbor_fn, model_energy_fn, False

    def initialize(self, positions):
        super().initialize(positions)
        # Materialise the pretrained parameters eagerly so the energy function
        # closes over concrete arrays (not created inside a jit trace).
        self._uma_params = self._uma_init(
            jax.random.PRNGKey(0), jnp.asarray(positions), self._template
        )
