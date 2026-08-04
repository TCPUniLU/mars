# MARS

> **M**achine-Learned Force Field Framework for **A**utomated
> Confo**r**mational **S**ampling, Vibrational Spectroscopy, and
> Microsolvation — JAX-native and GPU-accelerated.

[![Documentation](https://img.shields.io/badge/docs-tcpunilu.github.io%2Fmars-blue.svg)](https://tcpunilu.github.io/mars/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MARS bundles three workflows that share a JAX backend, a pluggable
potential layer (SO3LR, MACE, dxtb, NCI, plus harmonic/LJ for testing),
and an automatic-configuration system:

- **Conformational sampling** — RMSD metadynamics + rotamer MD + optional
  genetic crossing, with MTD-only and MD-only variants
- **IR spectroscopy** — analytical Hessian (autodiff or finite differences)
  or MD-based dipole autocorrelation, with peak allocation to functional
  groups
- **Explicit solvation** — layer-by-layer shell construction with optional
  layerwise / global optimization, ~30 built-in solvents

---

## Install

Requires **Python ≥ 3.12**.

```bash
git clone https://github.com/TCPUniLU/mars.git
cd mars
pip install -e .            # CPU JAX; add the GPU wheel below
```

Pick the JAX backend for your hardware, then add the ML potential(s) you need:

```bash
pip install -U "jax[cuda12]"          # NVIDIA GPU (CUDA 12); or jax[cuda13], or plain jax for CPU

# SO3LR (default, recommended) — installed from source:
# v2 ships so3lr-s / so3lr-m / so3lr-l plus the legacy v1 model.
pip install -e ".[so3lr]"
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr && pip install . && cd ..
# pick a model at run time with --so3lr-model {so3lr-s,so3lr-m,so3lr-l,so3lr_v1}

pip install -e ".[mace]"              # MACE   (also: pip install mace-jax)
pip install "dxtb[libcint]>=0.4.0"    # dxtb (GFN-xTB); see the docs for the numpy note
```

Full instructions (GPU CUDA 12/13, the dxtb ↔ JAX NumPy conflict and how to
solve it, conda `environment.yml`, troubleshooting) are in the
[documentation](https://tcpunilu.github.io/mars/install/).

---

## First run

```bash
mars input.xyz --quick
```

That command auto-detects the topology, picks MTD parameters from the
molecular flexibility, runs an MTD + rotamer-MD sampling cycle, and
writes the unique low-energy conformers to `auto_final_ensemble.xyz`.

See the [Quickstart](https://tcpunilu.github.io/mars/quickstart/) for a
complete walkthrough with a methane XYZ.

---

## Documentation

| | |
|---|---|
| 📘 **[Documentation site](https://tcpunilu.github.io/mars/)** | Full docs, hosted on GitHub Pages |
| 🚀 [Quickstart](https://tcpunilu.github.io/mars/quickstart/) | 5-minute first run |
| 📚 [User guide](https://tcpunilu.github.io/mars/guide/) | Concepts for each tool |
| 🧪 [Examples](https://tcpunilu.github.io/mars/examples/) | Copy-pasteable recipes |
| ⚙️ [Config files](https://tcpunilu.github.io/mars/config/overview/) | TOML `--config FILE` reference |
| 💻 [CLI reference](https://tcpunilu.github.io/mars/cli/overview/) | Every flag, organized by subcommand |
| 🐍 [Python API](https://tcpunilu.github.io/mars/api/) | Auto-generated from docstrings |

The four subcommands at a glance:

```bash
mars input.xyz                            # conformational search (default)
mars ir input.xyz --plot                  # IR spectrum
mars optimize input.xyz --fmax 0.001      # local minimum
mars solvation solute.xyz --solvent water --padding 5.0
```

---

## Testing

```bash
pip install -e .[dev]
pytest -m "not requires_so3lr and not requires_mace and not requires_dxtb"
```

The harmonic-potential test suite runs in ~2 minutes on a single CPU
and needs no ML weights.

---

## Extending MARS — bring your own potential

MARS is **model-agnostic**: any energy model can be plugged in with a single
`@register_potential` decorator and used immediately via `--potential NAME`.

```python
from mars import register_potential, PotentialWrapper

@register_potential("my_potential")
class MyPotential(PotentialWrapper):
    def _build_energy_fn(self):
        def energy_fn(positions, **kwargs):
            ...   # return a scalar energy in eV
        return energy_fn
```

**We're happy to help you wire in your own force field / ML potential** —
neighbour lists, partial charges for IR, `vmap` batching, registration. Open an
issue or email us and we'll help (and happily link or upstream it). See the
[potentials guide](https://tcpunilu.github.io/mars/guide/potentials/).

---

## Citation

MARS is described in a preprint of the same title — please cite it:

```bibtex
@article{mars2026,
  title   = {MARS: Machine-Learned Force Field Framework for Automated Conformational Sampling, Vibrational Spectroscopy, and Microsolvation},
  author  = {Su{\'a}rez-Dou, Sergio and Gallegos, Miguel and Tkatchenko, Alexandre},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/TCPUniLU/mars}
}
```

For the full list of methods, ML potentials, and reference data to cite,
see the [Citation page](https://tcpunilu.github.io/mars/about/citation/).

---

## License

MIT License — see [LICENSE](LICENSE). Developed in the Theoretical Chemical
Physics group at the University of Luxembourg, supported by the ERC grant
FITMOL.

## Support & contact

Questions, bug reports, and help integrating a new potential are all welcome:

- Issues / discussions: <https://github.com/TCPUniLU/mars/issues>
