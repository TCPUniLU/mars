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

Recent JAX (≥ 0.4.26, currently 0.10) **requires NumPy ≥ 2**. Choose the wheel
that matches your CUDA driver:

```bash
pip install -U "jax[cuda12]"      # NVIDIA GPU, CUDA 12.x   (most common)
pip install -U "jax[cuda13]"      # NVIDIA GPU, CUDA 13.x
pip install -U "jax"              # CPU only  (works, but not recommended)
```

- Check your driver with `nvidia-smi` (top-right shows the max CUDA version).
- The CUDA wheels bundle the needed CUDA libraries — you do **not** need a
  system CUDA toolkit, only a recent NVIDIA driver.
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
    pip install /path/to/so3lr_dev    # e.g. ../so3lr_dev-main
    ```

With the developing package, select a bundled v2 model with `--so3lr-model`:
`so3lr-s` (small), `so3lr-m` (medium), `so3lr-l` (large), or `so3lr_v1` (legacy
v1, **default**). A filesystem path to a custom / fine-tuned model workdir also
works. The v2 models are still under active development, so the v1 default
remains the safe choice for production.

### MACE

```bash
pip install -e ".[mace]"           # mace-torch
pip install mace-jax               # JAX-native inference (recommended)
```

MACE runs **JAX-native**: a pretrained foundation model is converted from its
PyTorch checkpoint once (PyTorch is only needed for that first conversion) and
cached under `~/.cache/mars/mace_jax`; later runs reload the JAX weights
directly. Select the family/size with `--mace-foundation {off,mp,anicc,omol}`
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
    pip install -U "jax[cuda12]"
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
# then add the GPU wheel and the SO3LR source install:
pip install -U "jax[cuda12]"
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr && pip install . && cd ..
```

| Extra | Pulls in |
|-------|----------|
| `[viewer]` | matplotlib (3D viewer / plots) |
| `[cuda12]` / `[cuda13]` | `jax[cuda12]` / `jax[cuda13]` |
| `[so3lr]` | ase, h5py (SO3LR itself is a source install) |
| `[mace]` | mace-torch (`pip install mace-jax` for JAX inference) |
| `[dxtb]` | `dxtb[libcint]>=0.4` |
| `[all]` | viewer + so3lr helpers + mace + dxtb |
| `[dev]` | pytest, ruff, black |
| `[docs]` | mkdocs-material, mkdocstrings, glightbox |

## Conda environment file

An [`environment.yml`](https://github.com/TCPUniLU/mars/blob/main/environment.yml)
is shipped at the repo root. The base file is CPU + viewer; GPU and the
individual potentials are commented blocks you uncomment:

```bash
conda env create -f environment.yml
conda activate mars
# GPU + SO3LR source install as above
```

---

## Python compatibility

MARS requires Python **3.12 or newer**. Python **3.12** is recommended (it
matches the SO3LR / MACE-JAX wheels best).

## Verifying the install

```bash
# tests that need no ML-potential weights (~290 tests, ~2 min on CPU)
pytest -m "not requires_so3lr and not requires_mace and not requires_dxtb"
```
