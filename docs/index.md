---
hide:
  - toc
---

# ![MARS](assets/logo.png){ .home-logo .off-glb }

A JAX-native, GPU-accelerated toolkit for conformational search, IR
spectroscopy, and explicit solvation — built on one pluggable potential layer
(SO3LR, MACE, dxtb…) that auto-tunes its metadynamics from molecular flexibility.

[Install :material-arrow-right:](install.md){ .md-button .md-button--primary }
[Quickstart :material-arrow-right:](quickstart.md){ .md-button }
[Examples :material-arrow-right:](examples/index.md){ .md-button }

---

<div class="feature-gifs" markdown="0">
<div class="feat-card">
<a class="feat-link" href="guide/conformer_search/" aria-label="Conformational search"></a>
<div class="feat-media"><img class="off-glb" src="assets/conf_sample.gif" alt="Conformational search" loading="lazy"></div>
<div class="feat-body"><span class="feat-title">Conformational search</span><span class="feat-desc">RMSD-biased metadynamics + rotamer MD + genetic crossing, auto-tuned.</span></div>
</div>
<div class="feat-card">
<a class="feat-link" href="guide/ir_spectroscopy/" aria-label="IR spectroscopy"></a>
<div class="feat-media"><img class="off-glb" src="assets/viewer/ir_viewer.gif" alt="IR spectroscopy" loading="lazy"></div>
<div class="feat-body"><span class="feat-title">IR spectroscopy</span><span class="feat-desc">Hessian (autodiff / FD) or MD dipole autocorrelation, with mode assignment.</span></div>
</div>
<div class="feat-card">
<a class="feat-link" href="guide/solvation/" aria-label="Explicit solvation"></a>
<div class="feat-media"><img class="off-glb" src="assets/solvation/barostat_solvation.gif" alt="Explicit solvation" loading="lazy"></div>
<div class="feat-body"><span class="feat-title">Explicit solvation</span><span class="feat-desc">Layer-by-layer packing or a moving-wall barostat droplet, ~30 solvents.</span></div>
</div>
<div class="feat-card">
<a class="feat-link" href="guide/nci_mode/" aria-label="NCI mode"></a>
<div class="feat-media"><img class="off-glb" src="assets/viewer/nci_viewer.gif" alt="NCI mode" loading="lazy"></div>
<div class="feat-body"><span class="feat-title">NCI mode</span><span class="feat-desc">Spherical confinement for host&ndash;guest &amp; multi-fragment complexes.</span></div>
</div>
</div>

---

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } **Structure optimization**

    ---

    LBFGS / FIRE / GD / damped-Newton search, with `vmap` batch
    optimization over entire ensembles.

    [:octicons-arrow-right-24: Guide](guide/optimization.md) ·
    [Recipes](examples/optimize_recipes.md)

-   :material-cube-outline:{ .lg .middle } **Pluggable potentials**

    ---

    SO3LR, MACE, dxtb, NCI confinement, plus harmonic / LJ test
    potentials. One-line decorator to register your own.

    [:octicons-arrow-right-24: Guide](guide/potentials.md) ·
    [API](api/potentials.md)

-   :material-file-cog-outline:{ .lg .middle } **Config files**

    ---

    Drive every subcommand from a single TOML file — defaults,
    frozen-atom constraints, per-tool sections. CLI flags override.

    [:octicons-arrow-right-24: Overview](config/overview.md)

</div>
