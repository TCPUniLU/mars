# ML potentials

Every ML potential in MARS runs in JAX: forces and Hessians come from
automatic differentiation, and replicas are batched with `vmap`. Pick a model
from the tabs below. The [overview](index.md) compares them all.

=== "SO3LR"

    **Default potential.** SO3LR covers organic and biomolecular systems and
    includes explicit long-range electrostatics and dispersion. It provides
    partial charges, so IR intensities work.

    ```bash
    mars input.xyz                                          # so3lr-1
    mars input.xyz --so3lr-model so3lr-2-m                  # v2 medium
    mars input.xyz --so3lr-model so3lr-2-m --charge -1 --spin 2   # radical anion (v2 only)
    ```

    | `--so3lr-model` | Model | Requires |
    |-----------------|-------|----------|
    | `so3lr-1` *(default)* | v1 | stable or developing package |
    | `so3lr-2-s` / `so3lr-2-m` / `so3lr-2-l` | v2 small / medium (recommended) / large | developing package (`so3lr` ≥ 0.2) |
    | *(a path)* | custom or fine-tuned workdir | developing package |

    | Option | Meaning |
    |--------|---------|
    | `--charge`, `--spin` | Total charge; spin multiplicity. Only the **v2** models are spin-aware (MARS passes M − 1 unpaired electrons). v1 is closed-shell only and refuses `--spin` ≠ 1 |
    | `--lr-cutoff Å` | Long-range cutoff (1000 Å gas phase) |
    | `--so3lr-backend jaxmd` | Use jax-md's port (`jax_md._nn.so3lr`) instead of the `so3lr` package. It ships the v1 weights (closed-shell only), and its long-range cutoff is fixed at 12 Å. |

    **Install:** from source, see [Installation](../../install.md#so3lr-default-recommended).
    The pre-release names `so3lr_v1`, `so3lr`, `so3lr-s/m/l` still work but
    are deprecated.

=== "MACE"

    The MACE foundation models, converted once from the PyTorch checkpoint to
    JAX and cached under `~/.cache/mars/mace_jax`.

    ```bash
    mars input.xyz --potential mace --mace-foundation off --mace-model small
    mars input.xyz --potential mace --mace-foundation off24      # medium only
    mars input.xyz --potential mace --mace-foundation omol --charge 1 --spin 2
    ```

    | `--mace-foundation` | Model | Charge / spin | Default backend |
    |---------------------|-------|---------------|-----------------|
    | `off` | MACE-OFF23 (small / medium / large) | — | JAX (mace-jax) |
    | `off24` | MACE-OFF24 (medium) | — | JAX (mace-jax) |
    | `mp` | MACE-MP-0 (small / medium / large, `medium-mpa-0`) | — | JAX (mace-jax) |
    | `anicc` | MACE-ANI-CC | — | JAX (mace-jax) |
    | `omol` | MACE-OMol (`extra_large`) | ✅ / ✅ | **PyTorch** (see below) |

    `--mace-backend` picks the implementation:

    | Value | Implementation |
    |-------|----------------|
    | `auto` *(default)* | `mace-jax`; `torch` for OMol |
    | `mace-jax` | ACEsuit mace-jax conversion (the MACE developers' JAX port) |
    | `jaxmd` | jax-md's MACE interface (`jax_md._nn.mace`, PR #390) |
    | `torch` | mace-torch ASE calculator: the reference, but no analytical Hessian |

    !!! warning "MACE-OMol in pure JAX"
        `--mace-backend mace-jax` and `jaxmd` accept OMol but print a warning:
        as of October 2026 neither conversion reproduces mace-torch. The
        ACEsuit conversion is about 0.9 eV off; jax-md's pinned mace-jax cannot
        build the OMol gates. `auto` keeps OMol on PyTorch, so IR falls back to
        finite-difference Hessians.

    MACE does not expose partial charges, so it gives frequencies but no IR
    intensities. **Install:** `pip install -e ".[mace]"` and
    `pip install "git+https://github.com/ACEsuit/mace-jax"`.

=== "UMA"

    FAIR-Chem's universal model, through jax-md's pure-JAX port.

    ```bash
    mars input.xyz --potential uma --uma-model uma-s-1p2 --uma-task omol --charge 0 --spin 1
    ```

    | Option | Values |
    |--------|--------|
    | `--uma-model` | `uma-s-1p1`, `uma-s-1p2` *(default)*, `uma-m-1p1`, or a checkpoint path |
    | `--uma-task` | `omol` *(molecules, default)*, `omat`, `oc20`, `odac`, `omc` |
    | `--charge`, `--spin` | Used only by the `omol` task. Other tasks refuse non-default values. |

    The weights are gated: accept the licence at
    <https://huggingface.co/facebook/UMA>, then run `huggingface-cli login`.
    The first run converts the checkpoint, which needs `torch`.
    **Install:** `pip install -e ".[uma]"`.

=== "ORB"

    Orb-v3 conservative, trained on OMol25, through jax-md's port.

    ```bash
    mars input.xyz --potential orb --charge -1 --spin 1
    ```

    Charge- and spin-aware. Spin is the multiplicity, as in orb-models.
    **Install:** jax-md main, plus `pip install -e ".[jaxmd-ml]"` and the
    weights (see [jax-md model weights](../../install.md#jax-md-model-weights)).

=== "AIMNet2"

    AIMNet2 through jax-md's port: neutral and charged closed-shell
    molecules, with explicit all-pairs Coulomb in free space.

    ```bash
    mars input.xyz --potential aimnet2 --charge -1
    ```

    Charge-aware; refuses `--spin` ≠ 1. Its equilibrated partial charges drive
    IR intensities.

    !!! note "Fix applied by MARS"
        jax-md's DFT-D3 dispersion underflows to zero for atoms whose
        coordination number is far from every D3 reference, such as
        hypervalent S and P in sulfonates and phosphates. That gave energy
        errors of 0.1–0.2 eV and NaN forces. MARS uses the same stabilised C6
        interpolation as the original `aimnet` package, and now matches it to
        10⁻⁷ eV.

=== "AceFF"

    AceFF 1.1 and 2.0 (TensorNet models from Acellera) through jax-md's port,
    for drug-like small molecules.

    ```bash
    mars input.xyz --potential aceff                          # AceFF 2.0
    mars input.xyz --potential aceff --ml-model aceff-jax-1.1
    ```

    !!! note "Electrostatics of AceFF 2.0"
        jax-md's AceFF 2.0 port has a 12 Å reaction-field Coulomb term with
        a solvent dielectric of 78.3 built in. The AceFF checkpoint's own
        default, used by torchmd-net, is plain Coulomb in vacuum, so MARS
        restores it. In Python, `get_potential("aceff", ..., coulomb_cutoff=12.0)`
        reproduces upstream jax-md instead.

    Charge-aware; refuses `--spin` ≠ 1. For IR intensities MARS uses
    effective atomic charges: the per-layer charge channels averaged with the
    weights of AceFF's own Coulomb term. They sum to the total charge to
    within about 10⁻⁴ e.

=== "ANI"

    ANI-2x (8-model ensemble) through jax-md's port. Neutral, closed-shell
    molecules of H, C, N, O, F, S and Cl only. MARS refuses other elements,
    charges and spins.

    ```bash
    mars input.xyz --potential ani
    mars input.xyz --potential ani --ml-model ani2x-jax-model0     # single member
    ```

    ANI has no partial charges, so it gives no IR intensities. MARS uses
    TorchANI's Hartree→eV constant (27.211386024367243), so energies match
    TorchANI exactly; jax-md itself uses the CODATA-2018 value, a relative
    difference of 8×10⁻⁹.

## Charge and spin

MARS always takes `--charge Q` and `--spin M`, where M is the multiplicity
2S+1, and converts them to each model's convention. The charge, multiplicity
and electron count are checked first; a neutral radical, for example, needs
`--spin 2`.

| Potential | Charge | Spin input | What the model receives |
|-----------|:------:|:----------:|-------------------------|
| `so3lr` v2 (`so3lr-2-s/m/l`) | ✅ | ✅ | unpaired electrons, M − 1 |
| `so3lr` v1, `so3lr_jaxmd` | ✅ | — | refuses M ≠ 1 (the v1 spin input was never trained) |
| `mace` (OMol) | ✅ | ✅ | multiplicity M |
| `mace` (OFF, MP, ANI-CC) | — | — | refuses M ≠ 1; warns that a charge is ignored |
| `uma` (`omol` task) | ✅ | ✅ | multiplicity M |
| `uma` (other tasks) | — | — | charge 0, spin 0 ("unset"), as FAIR-Chem does |
| `orb` | ✅ | ✅ | multiplicity M |
| `aimnet2`, `aceff` | ✅ | — | refuses M ≠ 1 |
| `ani` | — | — | refuses Q ≠ 0 or M ≠ 1 |
| `dxtb` | ✅ | ✅ | unpaired electrons, M − 1 |
| classical force fields | from parameters | — | refuses M ≠ 1; `--charge` is checked against the parameters |

## Supported elements

Each model is checked against the elements it was trained on before anything
runs. An unsupported element stops with a message that names it and lists
what the model covers.

| Potential | Elements |
|-----------|----------|
| `so3lr` v1, `so3lr_jaxmd` | H, C, N, O, F, P, S, Cl |
| `so3lr` v2 | the QCML coverage: Z = 1–85 except He, Ne, Ar, Rb, Sr, Ba ([Ganscha et al., *Sci. Data* 12, 406 (2025)](https://doi.org/10.1038/s41597-025-04720-7), Fig. 1b) |
| `mace` | read from the model (e.g. MACE-OFF: H, C, N, O, F, P, S, Cl, Br, I; MACE-MP: 89 elements) |
| `mace` OMol, `uma` (`omol` task), `orb` | the 83 OMol25 elements (H–Bi) |
| `uma` (other tasks) | Z = 1–100 |
| `aimnet2` | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |
| `aceff` | H, B, C, N, O, F, Si, P, S, Cl, Br, I |
| `ani` | H, C, N, O, F, S, Cl |
| `dxtb` | H–Rn (Z = 1–86) |

The charge, multiplicity and electron count are also checked for
consistency; an odd-electron molecule declared as a singlet is refused, for
example.

## IR intensities

IR intensities need the molecular dipole. MARS builds it from the partial
charges a potential provides (μ = Σ qᵢ rᵢ). For AMOEBA it also includes the
permanent and induced atomic dipoles.

| Potential | Source of the dipole |
|-----------|----------------------|
| `so3lr`, `so3lr_jaxmd` | SO3LR partial charges |
| `aimnet2` | AIMNet2 equilibrated charges |
| `aceff` | AceFF Coulomb-weighted charges |
| `dxtb` | Mulliken charges |
| `amber`, `charmm`, `opls` | Fixed force-field charges |
| `amoeba` | Monopoles plus permanent and induced dipoles |
| `mace`, `uma`, `orb`, `ani` | none: frequencies only |

## jax-md models: installation

`uma`, `orb`, `ani`, `aimnet2`, `aceff` and `--so3lr-backend jaxmd` come from
[jax-md](https://github.com/jax-md/jax-md). UMA's port is already in jax-md
0.2.29; the others need jax-md main:

```bash
pip install "jax-md @ git+https://github.com/jax-md/jax-md.git"
pip install -e ".[jaxmd-ml]"          # equinox
python -m jax_md._nn.weights orb      # or ani, aimnet2, aceff, so3lr
```

If the last command fails with HTTP 404, see
[jax-md model weights](../../install.md#jax-md-model-weights).

These ports are new upstream. Until a model passes the MARS validation
suite, it prints a one-time "experimental" warning; see
[Validation and credits](credits.md).
