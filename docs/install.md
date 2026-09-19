# Installation

MARS is a JAX program. A typical install has three layers:

1. **JAX** matched to your hardware (CUDA 12, CUDA 13, or CPU).
2. **MARS** itself (`pip install -e .`).
3. One or more **ML potentials** (SO3LR, MACE, dxtb) — all optional.

!!! tip "Use an isolated environment"
    Always install into a fresh `conda`/`venv` environment. The ML potentials
    pull in large, sometimes conflicting dependency trees (see
    [Dependency conflicts](#dependency-conflicts-dxtb-vs-the-jax-stack)).

---

## 1. Create an environment

=== "conda"

    ```bash
    conda create -n mars python=3.12
    conda activate mars
    ```

=== "venv"

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

A ready-made conda file is provided — see [Conda environment file](#conda-environment-file).

## 2. Install JAX (pick one)

Recent JAX (≥ 0.4.26; 0.10.x is the tested line) **requires NumPy ≥ 2**. Choose the wheel
that matches your CUDA driver:

```bash
pip install -U "jax[cuda12]<0.11"   # NVIDIA GPU, CUDA 12.x   (most common)
pip install -U "jax[cuda13]<0.11"   # NVIDIA GPU, CUDA 13.x
pip install -U "jax<0.11"           # CPU only  (works, but not recommended)
```

!!! warning "Keep JAX below 0.11"
    SO3LR requires `jax<0.11` (JAX 0.11.1 has an XLA:CPU regression that hangs
    its mesh scatter). Install JAX **with the bound**: if you install an
    unbounded `jax[cuda13]` (0.11.x) first and SO3LR afterwards, pip downgrades
    `jax`/`jaxlib` but can leave the CUDA plugin wheels on 0.11 — a version
    mismatch that breaks GPU support. The `[so3lr]`, `[cuda12]` and `[cuda13]`
    extras carry this bound.

- Check your driver with `nvidia-smi` (top-right shows the max CUDA version).
- The CUDA wheels bundle the needed CUDA libraries — you do **not** need a
  system CUDA toolkit, only a recent NVIDIA driver.
- A CUDA 13 driver works with `jax[cuda13]` for SO3LR *and* MACE; there is no
  need to fall back to CUDA 12 or install an adapter. `mace-torch` pulls in its
  own PyTorch CUDA wheels, which coexist with the JAX ones in the same
  environment.
- CPU-only JAX runs everything correctly but is **much** slower; fine for the
  viewer, small molecules, and testing. Force CPU at runtime any time with
  `JAX_PLATFORMS=cpu`.

## 3. Install MARS

```bash
git clone https://github.com/TCPUniLU/mars.git
cd mars
pip install -e .
```

This installs JAX-MD, jaxopt, NumPy, SciPy, tqdm and the `mars` console script.
Verify:

```bash
mars --help
python -c "import mars; print(mars.__version__)"
```

The 3D **viewer** needs matplotlib (no JAX/GPU required):

```bash
pip install -e ".[viewer]"
```

---

## 4. ML potentials (optional)

You only need the backends you intend to use. They are selected at runtime with
`--potential {so3lr,mace,dxtb}`.

### SO3LR (default, recommended)

[SO3LR](https://github.com/general-molecular-simulations/so3lr) is the default
potential and provides the partial charges used for IR intensities. It is
installed from source:

```bash
pip install -e ".[so3lr]"          # ase, h5py helpers
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr && pip install . && cd ..
```

**Two SO3LR packages are supported** and MARS auto-detects which is installed:

- **Stable `so3lr`** (the command above, v0.1.x) — ships the **v1** model only.
  `--so3lr-model` is ignored (MARS warns and uses v1).
- **Developing package** (`so3lr_dev`, ≥ 0.2) — adds the **v2** model registry.
  Install it in place of the stable one (a separate environment is convenient):

    ```bash
    pip install -U "orbax-checkpoint>=0.12"   # older orbax breaks on JAX >= 0.10
    pip install /path/to/so3lr_dev            # e.g. ../so3lr_dev-main
    ```

    `so3lr_dev` pins `flax<0.12.9`; the `[so3lr]` extra above already carries
    the matching `flax` and `orbax-checkpoint` constraints.

With the developing package, select a bundled v2 model with `--so3lr-model`:
`so3lr-2-s` (small), `so3lr-2-m` (medium), `so3lr-2-l` (large), or `so3lr-1`
(legacy v1, **default**). A filesystem path to a custom / fine-tuned model
workdir also works. The v2 models are still under active development, so the
v1 default remains the safe choice for production. The pre-release names
(`so3lr_v1`, `so3lr`, `so3lr-s`, `so3lr-m`, `so3lr-l`) still work but are
deprecated.

### MACE

```bash
pip install -e ".[mace]"                              # mace-torch + compatible flax
pip install "git+https://github.com/ACEsuit/mace-jax"  # JAX-native inference (required)
```

!!! note "`mace-jax` is not on PyPI"
    `pip install mace-jax` fails with *No matching distribution*; install it
    from GitHub as above. It needs **flax ≥ 0.12** (the `[mace]` extra pins
    `flax>=0.12,<0.12.9`, which is also what SO3LR accepts, so both potentials
    share one environment).

MACE runs **JAX-native**: a pretrained foundation model is converted from its
PyTorch checkpoint once (PyTorch is only needed for that first conversion) and
cached under `~/.cache/mars/mace_jax`; later runs reload the JAX weights
directly. Select the family/size with `--mace-foundation {off,off24,mp,anicc,omol}`
and `--mace-model`. MACE does not expose partial charges, so use SO3LR or dxtb
for IR intensities.

### dxtb (GFN1/GFN2-xTB)

```bash
pip install "dxtb[libcint]>=0.4.0"
```

`dxtb` is a **PyTorch-based** differentiable tight-binding code (it is *not* a
JAX package). The `libcint` extra (`tad-libcint`, Linux only) is highly
recommended for performance. Use it with `--potential dxtb --dxtb-method {gfn1,gfn2}`;
it provides Mulliken charges, so it can drive IR intensities (via
finite-difference Hessians).

---

## Dependency conflicts (dxtb vs. the JAX stack)

This is the one combination that needs care.

- **JAX ≥ 0.4.26 requires `numpy >= 2`** (SO3LR and MACE-JAX run on this stack).
- **Older `dxtb` (≤ 0.3.x) pinned `numpy < 2`** through its `tad-*` dependencies
  — directly incompatible with JAX's `numpy ≥ 2`. Installing it next to JAX makes
  pip try to *downgrade* NumPy and break JAX.

**How to solve it**

1. **Use `dxtb ≥ 0.4.0`** (recommended). It dropped the `numpy < 2` cap, so it
   coexists with the JAX stack. Install **dxtb first**, then JAX/MARS on top, so
   the JAX stack — which requires `numpy ≥ 2` — settles the shared NumPy pin
   last:

    ```bash
    pip install "dxtb[libcint]>=0.4.0"
    pip install -U "jax[cuda12]<0.11"
    pip install -e ".[so3lr]"     # + the so3lr source install above
    ```

    Installing JAX/MARS after dxtb lets pip resolve NumPy up to the `≥ 2`
    version the JAX stack needs (dxtb ≥ 0.4 accepts it) rather than dxtb pinning
    it down. Then sanity-check nothing broke:

    ```bash
    python -c "import jax, numpy; print('jax', jax.__version__, '| numpy', numpy.__version__)"
    mars ir input.xyz --potential dxtb --plot
    ```

2. **If you must use an old dxtb that pins `numpy < 2`**, keep it in its **own
   environment**, separate from the JAX-MLFF env:

    ```bash
    conda create -n mars-dxtb python=3.12
    conda activate mars-dxtb
    pip install -e . "dxtb[libcint]"
    ```

    Run dxtb jobs there and SO3LR/MACE jobs in your main `mars` env. (dxtb also
    pulls in PyTorch, a large independent stack — another reason isolation is
    convenient.)

---

## Install everything at once

`pip` extras for the pip-installable potentials:

```bash
pip install -e ".[all]"            # viewer + mace + dxtb + so3lr helpers
# then add the GPU wheel, mace-jax, and the SO3LR source install:
pip install -U "jax[cuda12]<0.11"  # or jax[cuda13]<0.11
pip install "git+https://github.com/ACEsuit/mace-jax"
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr && pip install . && cd ..
```

| Extra | Pulls in |
|-------|----------|
| `[viewer]` | matplotlib (3D viewer / plots) |
| `[cuda12]` / `[cuda13]` | `jax[cuda12]<0.11` / `jax[cuda13]<0.11` |
| `[so3lr]` | ase, h5py, `jax<0.11`, `flax<0.12.9`, `orbax-checkpoint>=0.12` (SO3LR itself is a source install) |
| `[mace]` | mace-torch, `flax>=0.12,<0.12.9` (`mace-jax` is a separate GitHub install) |
| `[dxtb]` | `dxtb[libcint]>=0.4` |
| `[all]` | viewer + so3lr helpers + mace + dxtb (still add `mace-jax` and the SO3LR source install) |
| `[dev]` | pytest, ruff, black |
| `[docs]` | mkdocs-material, mkdocstrings, glightbox |

## Conda environment file

An [`environment.yml`](https://github.com/TCPUniLU/mars/blob/main/environment.yml)
is shipped at the repo root. The base file is CPU + viewer; GPU and the
individual potentials are commented blocks you uncomment:

```bash
conda env create -f environment.yml
conda activate mars
# then, inside the env: GPU wheel, mace-jax and the SO3LR source install as above
```

---

## Python compatibility

MARS requires Python **3.12 or newer**. Python **3.12** is recommended (it
matches the SO3LR / MACE-JAX wheels best).

## Verifying the install

```bash
# tests that need no ML-potential weights (~430 tests, ~5 min)
pytest -m "not requires_so3lr and not requires_mace and not requires_dxtb"

# full suite, including the SO3LR tests (needs SO3LR installed; ~440 tests)
pytest
```

A quick end-to-end check of each potential you installed (writes a tiny
`water.xyz` and optimizes it):

```bash
printf '3\nwater\nO 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\n' > water.xyz
mars optimize water.xyz --potential so3lr
mars optimize water.xyz --potential mace --mace-foundation off --mace-model small
python -c "import jax; print(jax.devices())"     # should list a CudaDevice on GPU
```

The first MACE run downloads the foundation model and converts it to JAX
(needs internet); later runs reuse `~/.cache/mars/mace_jax`.

## Tested versions

The SO3LR (`so3lr_dev` 0.2.0) + MACE combination was verified on Linux with an
NVIDIA driver reporting CUDA 13.0, Python 3.12:

| Package | Version |
|---------|---------|
| `jax` / `jaxlib` / `jax-cuda13-plugin` | 0.10.2 |
| `flax` | 0.12.8 |
| `orbax-checkpoint` | 0.12.4 |
| `numpy` | 2.5 |
| `mace-torch` / `mace-jax` | 0.3.16 / 0.2.0 (GitHub) |
| `torch` | 2.14 (`+cu130`) |

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `SO3LR is not installed or failed to import` with a chained `AttributeError: ... has no attribute 'DeviceLocalLayout'` | `orbax-checkpoint` older than 0.12 with JAX ≥ 0.10. `pip install -U "orbax-checkpoint>=0.12"`. |
| `module 'flax.nnx' has no attribute 'List'` (MACE) | flax older than 0.12. `pip install "flax>=0.12,<0.12.9"`. |
| `pip install mace-jax`: *No matching distribution* | Not on PyPI. `pip install "git+https://github.com/ACEsuit/mace-jax"`. |
| JAX falls back to CPU (`[CpuDevice(id=0)]`) | The matching `jax[cuda12]`/`jax[cuda13]` wheel is missing, or `JAX_PLATFORMS=cpu` is set. |
| `No matching distribution` / NumPy downgrade while installing dxtb | See [Dependency conflicts](#dependency-conflicts-dxtb-vs-the-jax-stack). |
