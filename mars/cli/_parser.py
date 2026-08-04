"""
Argument parser construction for MARS CLI.
"""

import argparse


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mars",
        description="""MARS — Machine-Learned Force Field Framework for Automated
Conformational Sampling, Vibrational Spectroscopy, and Microsolvation

A JAX-native, GPU-accelerated workflow for conformational search, IR
spectroscopy, and explicit solvation. Three subcommands plus the default
conformational-search mode share a pluggable potential layer and an
automatic-configuration system.

SUBCOMMANDS
  (default)   Conformational sampling — full workflow, MTD-only, or MD-only
  ir          IR spectroscopy — analytical Hessian (autodiff/FD) or MD-based
  optimize    Structure optimization — local minimum or stationary point
  solvation   Explicit solvation — layer-by-layer shell construction
  viewer      Interactive 3D viewer — molecule, IR results, or NCI wall

COMMON OPTIONS (available on every subcommand)
  --potential {so3lr,mace,dxtb,nci,harmonic,lj}
                    Potential to use (default: so3lr)
  --mace-foundation {mp,off,anicc,omol} · --mace-model NAME
                    MACE selector  (only with --potential mace)
  --dxtb-method {gfn1,gfn2}
                    dxtb method    (only with --potential dxtb)
  --so3lr-model NAME|PATH · --lr-cutoff Å
                    SO3LR model (so3lr-s/-m/-l/so3lr_v1 or path) and long-range cutoff
  --charge Q        Total molecular charge (default: 0.0)
  --float64         Use 64-bit precision (default: 32-bit; auto-promoted for IR)
  --cpu             Force CPU execution
  --not-parallel    Disable vmap parallelism (sequential batching)
  --config FILE     Load defaults from a TOML control file (CLI flags override)
  --log-file FILE · --log-level {DEBUG,INFO,WARNING,ERROR} · --debug
                    Logging controls

CONFORMATIONAL SEARCH (default subcommand)
  Modes:        --quick | --thorough | --mode {quick,normal,thorough}
  Workflow:     --mtd-only | --md-only | --genetic-crossing [--n-children N]
  Metadynamics: --mtd-cycles · --mtd-time PS · --mtd-dt FS
                --kpush KCAL · --alpha · --cvdump-fs · --kscal · --no-trial
  Rotamer MD:   --rotamer-structures · --rotamer-temps · --rotamer-time
  Pruning:      --ewin KCAL · --optlevel · --no-topology
  Optimizer:    --method {LBFGS,FIRE,GD,SP} · --fmax (final cycle, eV/Å)
                --maxiter · --max-stepsize
                FIRE:  --fire-dt-start · --fire-dt-max · --fire-n-min
  NCI mode:     --nci [--nci-buffer Å] [--nci-mode {wall,harmonic}]
                [--nci-max-dist Å]   (host–guest / ion pair / multi-fragment)
  Output:       -o ensemble.xyz · --save-trajectory · -T temp · --charge Q

IR (mars ir)
  Method:       (analytical Hessian by default)
                --fd-hessian [--fd-displacement Å]
                --md [--nvt-time --nve-time --md-timestep --dipole-save-fs]
                     [--n-replicas N --chop N --window {hann,hamming,...}]
  Geometry:     --no-optimize · --fmax (eV/Å, ignored if --no-optimize)
  Multi-conf:   --conformer N · --all-conformers
  Subsystem:    --atoms IDX        (partial Hessian / partial dipole on a
                                    selected atom subset, e.g. solute in
                                    explicit solvent; supports "0,1,2",
                                    "0-5", or "0-3,7,10-12")
  Output:       --plot [--plot-output] · --freq-range MIN MAX · --broadening
                --save-hessian · --save-structure · --mode-xyz · --no-analysis

OPTIMIZE (mars optimize)
  Task:         --task {minimize,stationary}
  Minimizer:    --method {LBFGS,GD,FIRE,SP} · --fmax · --maxiter · --max-stepsize
  FIRE:         --fire-dt-start · --fire-dt-max · --fire-n-min
  Batch:        --conformer N · --all-conformers · --output-prefix STR

SOLVATION (mars solvation)
  Library:      --solvent NAME|FILE  (~30 built-ins + aliases, or a path to your .xyz)
  Target:       (default: cover) | --padding Å | --n-molecules N | --layers N [N ...]
  Optimization: --opt-mode {none,after-all,layerwise} (default: layerwise)
                --method {FIRE,LBFGS,GD,HYBRID} · --freeze-mode · --fmax · --maxiter
  Placement:    --vdw-scale · --buffer · --n-candidates · --n-shells · --n-orient
                --refill-cycles · --seed
  Topology:     (post-relax integrity check, on by default)
                --no-topology-repair · --max-repair N · --topo-tolerance T
  Barostat:     --barostat (compress a spherical solvent droplet, then swap)
                --baro-max-force · --baro-k · --baro-wall-mode
                --baro-step · --baro-fine-step · --baro-inner-buffer
                --baro-radius · --baro-fill · --baro-max-steps
                --baro-md [--baro-md-temp --baro-md-time --baro-md-dt]
  Multi-conf:   --conformer N · --all-conformers · --solute-swap [--no-align]
                --n-solvations N · --relax-solute · --final-relax · --no-multistage
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES

  # Conformational search — auto-configured (recommended starting point)
  mars input.xyz                              # normal mode
  mars input.xyz --quick                      # fast, less thorough
  mars input.xyz --thorough --genetic-crossing
  mars input.xyz --mtd-only --mtd-cycles 3    # MTD only (no rotamers/crossing)
  mars input.xyz --md-only --mtd-time 20.0    # plain NVT MD (no bias)

  # Charged / non-covalent systems
  mars complex.xyz --charge -1.0 --ewin 4.0
  mars host_guest.xyz --nci --nci-mode wall --nci-buffer 3.0

  # Pick a different optimizer for the multilevel refinement cycles
  mars input.xyz --method FIRE                    # FIRE inside the workflow
  mars input.xyz --method GD --max-stepsize 0.05  # gradient descent
  mars input.xyz --method LBFGS --fmax 0.005 --maxiter 2000

  # IR spectroscopy
  mars ir input.xyz --plot                                # analytical Hessian
  mars ir input.xyz --fd-hessian --plot                   # finite differences
  mars ir input.xyz --md --n-replicas 4 --chop 10 --plot  # MD-based
  mars ir ensemble.xyz --all-conformers --plot            # multi-conformer

  # Optimization
  mars optimize input.xyz --fmax 0.001                    # local minimum
  mars optimize input.xyz --task stationary               # stationary point
  mars optimize ensemble.xyz --all-conformers             # batch (parallel)

  # Solvation
  mars solvation solute.xyz --solvent water --padding 5.0
  mars solvation solute.xyz --solvent EtOH --layers 4 8 --opt-mode layerwise
  mars solvation solute.xyz --solvent DMSO --n-molecules 30 --potential mace

  # Interactive viewer (molecule / IR results / NCI wall)
  mars viewer molecule.xyz                                # 3D molecule
  mars ir molecule.xyz --mode-xyz --plot && mars viewer --ir  # explore IR modes
  mars viewer complex.xyz --nci --nci-buffer 5.0          # preview NCI wall

  # Switching potentials
  mars input.xyz --potential so3lr --so3lr-model so3lr-l   # SO3LR v2 large
  mars input.xyz --potential mace --mace-foundation off --mace-model small
  mars input.xyz --potential dxtb --dxtb-method gfn2
  mars optimize input.xyz --potential harmonic            # toy potential (tests)

  # Load defaults from a TOML config file (CLI flags override)
  mars input.xyz --config run.toml
  mars optimize input.xyz --config run.toml               # [constraints] honored

  # Logging / diagnostics
  mars input.xyz --log-level DEBUG --log-file run.log
  mars input.xyz --debug                                  # DEBUG + show warnings

Tip: run any subcommand with --help for the full option reference.
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a TOML-style control file with default options "
        "(CLI flags override config values). See docs for format.",
    )

    # Subparsers for different modes (optional - default is conformational search)
    subparsers = parser.add_subparsers(
        dest="command", required=False, help="Command to execute (default: conformational search)"
    )

    # Build subparsers
    _build_ir_subparser(subparsers)
    _build_optimize_subparser(subparsers)
    _build_solvation_subparser(subparsers)
    _build_viewer_subparser(subparsers)

    # Add conformational search arguments to main parser
    _add_conformer_search_arguments(parser)

    return parser


def _build_ir_subparser(subparsers):
    """Build the 'ir' subparser with all its arguments."""
    ir_parser = subparsers.add_parser(
        "ir",
        help="Compute IR spectrum",
        description="Compute IR spectrum using SO3LR analytical Hessian or MD trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic IR calculation (analytical Hessian)
  mars ir input.xyz

  # Save detailed Hessian data (frequencies, force constants, Hessian matrix)
  mars ir input.xyz --save-hessian --hessian-output hessian.txt

  # Save optimized structure used for Hessian calculation
  mars ir input.xyz --save-structure --structure-output optimized.xyz

  # Save both Hessian data and optimized structure
  mars ir input.xyz --save-hessian --save-structure

  # Save all normal modes in a single XYZ file (positions + eigenvectors)
  mars ir input.xyz --mode-xyz

  # Save normal modes with custom output filename
  mars ir input.xyz --mode-xyz --mode-output my_modes.xyz

  # MD-based IR with dipole autocorrelation (5 ps NVT + 25 ps NVE)
  mars ir input.xyz --md --plot

  # MD-based IR with custom times and dipole saving interval
  mars ir input.xyz --md --nvt-time 50 --nve-time 200 --md-timestep 1.0 --dipole-save-fs 5.0

  # MD-based IR with trajectory chopping for better statistics
  mars ir input.xyz --md --nve-time 100 --chop 10 --window hann

  # With plot output
  mars ir input.xyz --plot

  # Custom temperature and output
  mars ir input.xyz -t 400 -o spectrum.dat --plot --plot-output spectrum.png

  # Multi-frame XYZ: select specific conformer
  mars ir ensemble.xyz --conformer 2

  # Multi-frame XYZ: process all conformers
  mars ir ensemble.xyz --all-conformers --plot

  # Charged molecule with custom frequency range
  mars ir input.xyz --charge -1.0 --plot --freq-range 500 3500

  # Partial Hessian over solute atoms only (e.g. solute+solvent system)
  mars ir solvated.xyz --atoms 0-8 --plot           # solute = atoms 0..8
  mars ir solvated.xyz --atoms "0-2,15"             # noncontiguous subset
  mars ir solvated.xyz --atoms 0-8 --md --plot      # partial dipole in MD
        """,
    )

    # Config file
    ir_parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a TOML-style control file with default options "
        "(CLI flags override config values)",
    )

    # IR arguments
    ir_parser.add_argument("input", type=str, help="Input structure file (XYZ format)")
    ir_parser.add_argument(
        "--potential",
        type=str,
        default="so3lr",
        help="Potential to use: so3lr, mace, dxtb, harmonic, lj (default: so3lr)",
    )
    ir_parser.add_argument(
        "--mace-foundation",
        type=str,
        default="off",
        choices=["mp", "off", "anicc", "omol"],
        help="MACE foundation family: off (default), mp, anicc, omol",
    )
    ir_parser.add_argument(
        "--mace-model",
        type=str,
        default="small",
        help="MACE variant name: small (default), medium, large, medium-mpa-0, ... "
        "(omol only has extra_large); may also be a local path or URL",
    )
    ir_parser.add_argument(
        "--mace-cache-dir",
        type=str,
        default=None,
        help="Directory for converted JAX weights (default: ~/.cache/mars/mace_jax)",
    )
    ir_parser.add_argument(
        "--spin",
        type=float,
        default=1.0,
        help="Spin multiplicity for charge/spin-aware MACE models (omol); default 1",
    )
    ir_parser.add_argument(
        "--dxtb-method",
        type=str,
        default="gfn1",
        choices=["gfn1", "gfn2"],
        help="dxtb method: gfn1 (default) or gfn2",
    )
    ir_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="ir_spectrum.dat",
        help="Output file for IR spectrum (default: ir_spectrum.dat)",
    )
    ir_parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    ir_parser.add_argument(
        "--charge", type=float, default=0.0, help="Total molecular charge (default: 0.0)"
    )
    ir_parser.add_argument(
        "--so3lr-model",
        type=str,
        default="so3lr_v1",
        help="SO3LR model: so3lr-s | so3lr-m | so3lr-l | so3lr_v1, or a path to a custom/fine-tuned model (default: so3lr_v1)",
    )
    ir_parser.add_argument(
        "--lr-cutoff",
        type=float,
        default=1000.0,
        help="Long-range cutoff in Angstrom (default: 1000 for gas phase)",
    )
    ir_parser.add_argument(
        "--float64", action="store_true", help="Use float64 precision (default: float32)"
    )
    ir_parser.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: console only)"
    )
    ir_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    ir_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode: verbose logging"
    )
    ir_parser.add_argument(
        "--plot", action="store_true", help="Generate IR spectrum plot (requires matplotlib)"
    )
    ir_parser.add_argument(
        "--plot-output",
        type=str,
        default="ir_spectrum.png",
        help="Output file for IR plot (default: ir_spectrum.png)",
    )
    ir_parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=[400, 4000],
        metavar=("MIN", "MAX"),
        help="Frequency range for plot in cm\u207b\u00b9 (default: 400 4000)",
    )
    ir_parser.add_argument(
        "--broadening",
        type=float,
        default=10.0,
        help="Lorentzian broadening width in cm\u207b\u00b9 (default: 10.0)",
    )
    ir_parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for MD IR plot in data points (default: 2.0, 0 to disable)",
    )
    ir_parser.add_argument(
        "--conformer",
        type=int,
        default=None,
        help="For multi-frame XYZ, which conformer to use (0-indexed)",
    )
    ir_parser.add_argument(
        "--all-conformers",
        action="store_true",
        help="Compute IR for all conformers in multi-frame XYZ",
    )
    ir_parser.add_argument(
        "--atoms",
        type=str,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Restrict the IR calculation to a subset of atoms (0-based "
        "indices). Useful for picking out the solute in a solute+solvent "
        "system. For the Hessian path only the selected atoms are displaced "
        "(forces remain over the full system) → partial 3K×3K Hessian. For "
        "the MD path the dipole μ(t) = Σ q_i r_i is summed only over the "
        "selected atoms. Accepts CSV, range, or mixed forms, "
        "e.g. '0,1,2', '0-5', '0-3,7,10-12'. Multiple tokens are joined "
        "with commas before parsing.",
    )
    ir_parser.add_argument(
        "--output-prefix",
        type=str,
        default="conformer",
        help="Prefix for output files when using --all-conformers (default: conformer)",
    )
    ir_parser.add_argument(
        "--no-optimize",
        action="store_true",
        default=False,
        help="Skip geometry optimization before computing the Hessian. "
        "By default, the structure is pre-optimized (recommended for accurate frequencies).",
    )
    ir_parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion for the pre-Hessian geometry optimization in eV/\u00c5 "
        "(default: 0.01). Ignored when --no-optimize is set.",
    )
    ir_parser.add_argument(
        "--md",
        action="store_true",
        help="Compute IR from MD trajectory using dipole autocorrelation (includes charges)",
    )
    ir_parser.add_argument(
        "--nvt-time",
        type=float,
        default=5.0,
        help="NVT equilibration time in ps for MD-based IR (default: 5.0)",
    )
    ir_parser.add_argument(
        "--nve-time",
        type=float,
        default=25.0,
        help="NVE production time in ps for MD-based IR (default: 25.0)",
    )
    ir_parser.add_argument(
        "--md-timestep",
        type=float,
        default=0.5,
        help="MD timestep in fs for MD-based IR (default: 0.5)",
    )
    ir_parser.add_argument(
        "--dipole-save-fs",
        type=float,
        default=2.5,
        help="Save dipole moments every N fs for MD-based IR (default: 2.5)",
    )
    ir_parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="Save MD-based NVE trajectory to a specified HDF5 file (default: None)",
    )
    ir_parser.add_argument(
        "--chop",
        type=int,
        default=10,
        help="Split trajectory into N segments for averaging (improves statistics, default: 10)",
    )
    ir_parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["hann", "hamming", "blackman", "bartlett", "none"],
        help="Window function for IR autocorrelation (default: hann)",
    )
    ir_parser.add_argument(
        "--save-hessian",
        action="store_true",
        help="Save detailed Hessian data including frequencies, force constants, and Hessian matrix (Hessian approach only)",
    )
    ir_parser.add_argument(
        "--hessian-output",
        type=str,
        default="ir_hessian_data.txt",
        help="Output file for detailed Hessian data (default: ir_hessian_data.txt)",
    )
    ir_parser.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the optimized structure to XYZ file (Hessian approach only)",
    )
    ir_parser.add_argument(
        "--structure-output",
        type=str,
        default="optimized_structure.xyz",
        help="Output file for optimized structure (default: optimized_structure.xyz)",
    )
    ir_parser.add_argument(
        "--mode-xyz",
        action="store_true",
        help="Save all normal modes in a multi-frame XYZ file with positions and eigenvectors (Hessian approach only)",
    )
    ir_parser.add_argument(
        "--mode-output",
        type=str,
        default="normal_modes.xyz",
        help="Output file for normal modes XYZ (default: normal_modes.xyz)",
    )
    ir_parser.add_argument(
        "--analysis-output",
        type=str,
        default="mode_analysis.txt",
        help="Output file for mode analysis (default: mode_analysis.txt)",
    )
    ir_parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip vibrational mode classification and analysis",
    )
    ir_parser.add_argument(
        "--fd-hessian",
        action="store_true",
        help="Approximate Hessian by finite differences instead of analytical JAX auto-differentiation",
    )
    ir_parser.add_argument(
        "--fd-displacement",
        type=float,
        default=0.01,
        metavar="DISP",
        help="Displacement step in \u00c5 for finite-differences Hessian (default: 0.01)",
    )
    ir_parser.add_argument(
        "--n-replicas",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent MD replicas per conformer for MD-based IR "
        "(different random seeds; spectra are averaged). Default: 1",
    )
    ir_parser.add_argument(
        "--not-parallel",
        action="store_true",
        default=False,
        help="Force sequential execution when multiple replicas or conformers are "
        "requested for MD-based IR (default: parallel with vmap when >1 trajectory)",
    )

    return ir_parser


def _build_optimize_subparser(subparsers):
    """Build the 'optimize' subparser with all its arguments."""
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Structure optimization (LBFGS, GD, or stationary point)",
        description="Optimize molecular structures: find minima (LBFGS/GD) or stationary points (SP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic minimization (LBFGS)
  mars optimize input.xyz

  # Tight convergence minimization
  mars optimize input.xyz --fmax 0.001 --maxiter 500

  # Gradient Descent optimization
  mars optimize input.xyz --method GD

  # Stationary point search (damped Newton)
  mars optimize input.xyz --method SP

  # SP with custom trust radius
  mars optimize input.xyz --method SP --max-stepsize 0.1

  # Save optimized structure
  mars optimize input.xyz -o optimized.xyz

  # Multi-conformer XYZ: optimize specific conformer
  mars optimize ensemble.xyz --conformer 2

  # Multi-conformer XYZ: optimize all conformers
  mars optimize ensemble.xyz --all-conformers --output-prefix opt

  # Charged molecule optimization
  mars optimize input.xyz --charge -1.0 --fmax 0.005
        """,
    )

    # Config file
    opt_parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a TOML-style control file with default options "
        "(CLI flags override config values)",
    )

    # Optimize arguments
    opt_parser.add_argument("input", type=str, help="Input structure file (XYZ format)")
    opt_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="optimized.xyz",
        help="Output file for optimized structure (default: optimized.xyz)",
    )
    # Minimization parameters
    opt_min_group = opt_parser.add_argument_group("Minimization Parameters")
    opt_min_group.add_argument(
        "--task",
        type=str,
        default="minimize",
        choices=["minimize", "stationary"],
        help="Optimization task: 'minimize' finds a local energy minimum using --method "
        "(default), 'stationary' finds a stationary point via damped Newton "
        "(equivalent to --method SP).",
    )
    opt_min_group.add_argument(
        "--method",
        type=str,
        default="LBFGS",
        choices=["FIRE", "LBFGS", "GD", "SP"],
        help="Minimizer used when --task minimize: LBFGS (default, jaxopt), FIRE (jax-md), "
        "GD (gradient descent). 'SP' is equivalent to --task stationary.",
    )
    opt_min_group.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force/gradient convergence criterion in eV/\u00c5 (default: 0.01)",
    )
    opt_min_group.add_argument(
        "--max-stepsize",
        type=float,
        default=0.2,
        help="Maximum per-atom step in \u00c5: LBFGS step bound, GD learning rate, "
        "or SP trust radius (not used by FIRE; default: 0.2)",
    )

    # FIRE-specific parameters
    fire_group = opt_parser.add_argument_group("FIRE Parameters (only used with --method FIRE)")
    fire_group.add_argument(
        "--fire-dt-start", type=float, default=0.05, help="FIRE initial time step (default: 0.05)"
    )
    fire_group.add_argument(
        "--fire-dt-max",
        type=float,
        default=0.1,
        help="FIRE maximum time step — controls max atomic displacement per step (default: 0.1)",
    )
    fire_group.add_argument(
        "--fire-n-min",
        type=int,
        default=2,
        help="FIRE minimum positive-power steps before dt increase (default: 2)",
    )

    # General optimization parameters
    opt_gen_group = opt_parser.add_argument_group("General Optimization Parameters")
    opt_gen_group.add_argument(
        "--maxiter", type=int, default=1000, help="Maximum optimization iterations (default: 1000)"
    )

    # Multi-conformer options
    opt_conf_group = opt_parser.add_argument_group("Multi-Conformer Options")
    opt_conf_group.add_argument(
        "--conformer",
        type=int,
        default=None,
        help="For multi-frame XYZ, which conformer to optimize (0-indexed)",
    )
    opt_conf_group.add_argument(
        "--all-conformers", action="store_true", help="Optimize all conformers in multi-frame XYZ"
    )
    opt_conf_group.add_argument(
        "--output-prefix",
        type=str,
        default="optimized",
        help="Prefix for output files when using --all-conformers (default: optimized)",
    )

    # Potential options
    opt_pot_group = opt_parser.add_argument_group("Potential Options")
    opt_pot_group.add_argument(
        "--potential",
        type=str,
        default="so3lr",
        help="Potential to use: so3lr, mace, dxtb, harmonic, lj (default: so3lr)",
    )
    opt_pot_group.add_argument(
        "--mace-foundation",
        type=str,
        default="off",
        choices=["mp", "off", "anicc", "omol"],
        help="MACE foundation family: off (default), mp, anicc, omol",
    )
    opt_pot_group.add_argument(
        "--mace-model",
        type=str,
        default="small",
        help="MACE variant name: small (default), medium, large, medium-mpa-0, ... "
        "(omol only has extra_large); may also be a local path or URL",
    )
    opt_pot_group.add_argument(
        "--mace-cache-dir",
        type=str,
        default=None,
        help="Directory for converted JAX weights (default: ~/.cache/mars/mace_jax)",
    )
    opt_pot_group.add_argument(
        "--spin",
        type=float,
        default=1.0,
        help="Spin multiplicity for charge/spin-aware MACE models (omol); default 1",
    )
    opt_pot_group.add_argument(
        "--dxtb-method",
        type=str,
        default="gfn1",
        choices=["gfn1", "gfn2"],
        help="dxtb method: gfn1 (default) or gfn2",
    )
    opt_pot_group.add_argument(
        "--charge", type=float, default=0.0, help="Total molecular charge (default: 0.0)"
    )
    opt_pot_group.add_argument(
        "--so3lr-model",
        type=str,
        default="so3lr_v1",
        help="SO3LR model: so3lr-s | so3lr-m | so3lr-l | so3lr_v1, or a path to a custom/fine-tuned model (default: so3lr_v1)",
    )
    opt_pot_group.add_argument(
        "--lr-cutoff",
        type=float,
        default=1000.0,
        help="Long-range cutoff in Angstrom (default: 1000 for gas phase)",
    )

    # Computational options
    opt_comp_group = opt_parser.add_argument_group("Computational Options")
    opt_comp_group.add_argument(
        "--not-parallel",
        action="store_true",
        default=False,
        help="Optimize conformers sequentially instead of as a vmapped batch "
        "(default: parallel when --all-conformers gives >1 structure)",
    )
    opt_comp_group.add_argument(
        "--float64", action="store_true", help="Use float64 precision (default: float32)"
    )
    opt_comp_group.add_argument(
        "--cpu", action="store_true", help="Force CPU execution (disable GPU)"
    )

    # Logging options
    opt_log_group = opt_parser.add_argument_group("Logging Options")
    opt_log_group.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: console only)"
    )
    opt_log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    opt_log_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode: verbose logging"
    )

    return opt_parser


def _add_conformer_search_arguments(parser):
    """Add conformational search arguments to the main parser."""
    # Note: Input argument is NOT added here to avoid conflicts with subparsers
    # It will be handled manually in main() for the default conformational search mode

    # Search mode (always uses auto-configuration)
    mode_group = parser.add_argument_group("Search Mode (always auto-configured)")
    mode_select = mode_group.add_mutually_exclusive_group()
    mode_select.add_argument(
        "--quick",
        action="store_true",
        help="Quick search mode: fast, less thorough (auto-configured)",
    )
    mode_select.add_argument(
        "--thorough",
        action="store_true",
        help="Thorough search mode: slow, more exhaustive (auto-configured)",
    )
    mode_group.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["quick", "normal", "thorough"],
        help="Explicitly set search mode (default: normal). Overrides --quick/--thorough flags",
    )

    # Conformer MTD options
    conformer_mtd_group = parser.add_argument_group("Conformer MTD Options (autoconf only)")
    conformer_mtd_group.add_argument(
        "--n-conformer-starts",
        type=int,
        default=None,
        help="Number of diverse conformer starting points (default: auto-determined from mode)",
    )
    conformer_mtd_group.add_argument(
        "--exploration-time",
        type=float,
        default=None,
        help="Exploration phase MTD time in ps (default: 20\u00d7 standard, auto-determined)",
    )
    conformer_mtd_group.add_argument(
        "--conformer-time",
        type=float,
        default=None,
        help="Conformer MTD time per run in ps (default: 5\u00d7 standard, auto-determined)",
    )

    # Alternative workflows
    workflow_group = parser.add_argument_group("Alternative Workflows")
    workflow_group.add_argument(
        "--mtd-only",
        action="store_true",
        help="MTD sampling only (no rotamer MD or genetic crossing)",
    )
    workflow_group.add_argument(
        "--md-only",
        action="store_true",
        help="Plain MD sampling only (no metadynamics bias, rotamer MD, or genetic crossing)",
    )
    workflow_group.add_argument(
        "--nci",
        action="store_true",
        help="Non-covalent interaction mode: apply a spherical confinement potential "
        "in all MTD/MD runs and suppress multi-molecule warnings. Use for "
        "host-guest complexes, ion pairs, solute-solvent systems, or any "
        "non-covalently bound multi-molecule input.",
    )
    workflow_group.add_argument(
        "--nci-buffer",
        type=float,
        default=3.0,
        metavar="Å",
        help="Buffer distance (Å) between outermost atom and confinement wall "
        "(default: 3.0). Only used with --nci.",
    )
    workflow_group.add_argument(
        "--nci-mode",
        type=str,
        default="wall",
        choices=["wall", "harmonic"],
        help="Confinement potential type: 'wall' (steep polynomial, hard boundary) "
        "or 'harmonic' (quadratic penalty, soft boundary). Default: wall. "
        "Only used with --nci.",
    )
    workflow_group.add_argument(
        "--nci-max-dist",
        type=float,
        default=3.0,
        metavar="Å",
        help="Maximum allowed inter-fragment contact distance (Å) above the initial "
        "closest atom–atom contact between fragments (default: 3.0). Structures "
        "where fragments drift further apart are discarded at each optimization "
        "cycle. Only used with --nci.",
    )

    # MTD parameters
    mtd_group = parser.add_argument_group("Metadynamics Parameters")
    mtd_group.add_argument(
        "--mtd-cycles", type=int, default=2, help="Number of MTD cycles (default: 2)"
    )
    mtd_group.add_argument(
        "--mtd-time",
        type=float,
        default=10.0,
        help="MTD simulation time per cycle in ps (default: 10.0)",
    )
    mtd_group.add_argument(
        "--mtd-dt", type=float, default=0.5, help="MTD timestep in fs (default: 0.5)"
    )
    mtd_group.add_argument(
        "--kpush", type=float, default=20.0, help="MTD Gaussian height in kcal/mol (default: 20.0)"
    )
    mtd_group.add_argument(
        "--alpha", type=float, default=0.5, help="MTD Gaussian width parameter (default: 0.5)"
    )
    mtd_group.add_argument(
        "--cvdump-fs",
        type=float,
        default=50.0,
        help="Hill deposition interval in fs (default: 50.0)",
    )
    mtd_group.add_argument(
        "--no-trial",
        action="store_true",
        help="Skip trial MTD validation (only used with autoconf)",
    )
    mtd_group.add_argument(
        "--kscal",
        type=float,
        default=1.0,
        help="Global kpush scaling factor for autoconf (default: 1.0)",
    )
    mtd_group.add_argument(
        "--no-topology",
        action="store_true",
        help="Disable bond topology pruning to remove bond breaking (default: Enable)",
    )
    # General parameters
    general_group = parser.add_argument_group("General Parameters")
    general_group.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    general_group.add_argument(
        "--optlevel",
        type=int,
        default=None,
        choices=[-3, -2, -1, 0, 1, 2],
        help="Optimization level: -3 (loose) to 2 (tight) (default: 0, auto-configured in auto mode)",
    )
    general_group.add_argument(
        "--ewin",
        type=float,
        default=None,
        help="Energy window for pruning in kcal/mol (default: 10.0, auto-configured in auto mode)",
    )

    # Optimizer selection (used during the multilevel refinement cycles)
    opt_group = parser.add_argument_group("Optimizer Options")
    opt_group.add_argument(
        "--method",
        type=str,
        default="LBFGS",
        choices=["LBFGS", "FIRE", "GD", "SP"],
        help="Minimizer used in every optimization cycle of the workflow "
        "(initial relaxation + 3-cycle multilevel ladder + rotamer/genetic "
        "refinement). LBFGS (default, jaxopt), FIRE (jax-md), GD (gradient "
        "descent), SP (damped Newton stationary-point search). SP is unusual "
        "for conformer enumeration; prefer `mars optimize --task stationary` "
        "for true saddle searches.",
    )
    opt_group.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Force-convergence threshold in eV/Å for the FINAL optimization "
        "cycle only (overrides the value derived from --optlevel via FMAX_MAP). "
        "Coarse pre-cycles stay scaled at 50× and 10× this final value. "
        "Default: auto from --optlevel.",
    )
    opt_group.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Maximum optimizer iterations for the FINAL cycle only (default: 1000). "
        "Coarse pre-cycles use 10 and 100 iterations respectively.",
    )
    opt_group.add_argument(
        "--max-stepsize",
        type=float,
        default=None,
        help="Maximum step size in Å for the FINAL cycle (default: 0.2). "
        "Coarse pre-cycles use 1.0 Å and 0.5 Å respectively.",
    )

    # FIRE optimizer parameters (used when --method FIRE)
    fire_group = parser.add_argument_group("FIRE Optimizer Parameters (--method FIRE)")
    fire_group.add_argument(
        "--fire-dt-start", type=float, default=0.05, help="FIRE initial time step (default: 0.05)"
    )
    fire_group.add_argument(
        "--fire-dt-max",
        type=float,
        default=0.1,
        help="FIRE maximum time step — controls max atomic displacement per step (default: 0.1)",
    )
    fire_group.add_argument(
        "--fire-n-min",
        type=int,
        default=2,
        help="FIRE minimum positive-power steps before dt increase (default: 2)",
    )

    # Rotamer MD parameters
    rotamer_group = parser.add_argument_group("Rotamer MD Parameters")
    rotamer_group.add_argument(
        "--rotamer-structures",
        type=int,
        default=5,
        help="Number of structures for rotamer MD (default: 5)",
    )
    rotamer_group.add_argument(
        "--rotamer-temps", type=int, default=3, help="Number of temperature replicas (default: 3)"
    )
    rotamer_group.add_argument(
        "--rotamer-time", type=float, default=10.0, help="Rotamer MD time in ps (default: 10.0)"
    )

    # Genetic crossing
    genetic_group = parser.add_argument_group("Genetic Crossing")
    genetic_group.add_argument(
        "--genetic-crossing", action="store_true", help="Enable genetic crossing"
    )
    genetic_group.add_argument(
        "--n-children",
        type=int,
        default=20,
        help="Number of children for genetic crossing (default: 20)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        default="auto_final_ensemble.xyz",
        help="Output ensemble file (default: auto_final_ensemble.xyz)",
    )
    output_group.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: console only)"
    )
    output_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    output_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: verbose logging + show all package warnings",
    )
    output_group.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save MD trajectories as XYZ files (e.g. mtd_1_trajectory.xyz, refinement_mtd_1_trajectory.xyz)",
    )

    # Computational options
    compute_group = parser.add_argument_group("Computational Options")
    compute_group.add_argument(
        "--not-parallel",
        action="store_true",
        default=False,
        help="Disable vmap parallelism: run MTD steps, rotamer MD replicas, and "
        "structure optimizations sequentially (default: parallel when >1 replica)",
    )
    compute_group.add_argument(
        "--float64",
        action="store_true",
        help="Use 64-bit floating point precision (default: 32-bit)",
    )
    compute_group.add_argument(
        "--cpu", action="store_true", help="Force CPU execution (disable GPU)"
    )
    compute_group.add_argument(
        "--potential",
        type=str,
        default="so3lr",
        help="Potential to use: so3lr, mace, dxtb, harmonic, lj (default: so3lr)",
    )

    # MACE-specific options
    mace_group = parser.add_argument_group("MACE Options (--potential mace)")
    mace_group.add_argument(
        "--mace-foundation",
        type=str,
        default="off",
        choices=["mp", "off", "anicc", "omol"],
        help="MACE foundation family: off (default), mp, anicc, omol",
    )
    mace_group.add_argument(
        "--mace-model",
        type=str,
        default="small",
        help="MACE variant name: small (default), medium, large, medium-mpa-0, ... "
        "(omol only has extra_large); may also be a local path or URL",
    )
    mace_group.add_argument(
        "--mace-cache-dir",
        type=str,
        default=None,
        help="Directory for converted JAX weights (default: ~/.cache/mars/mace_jax)",
    )
    mace_group.add_argument(
        "--spin",
        type=float,
        default=1.0,
        help="Spin multiplicity for charge/spin-aware MACE models (omol); default 1",
    )

    # dxtb-specific options
    dxtb_group = parser.add_argument_group("dxtb Options (--potential dxtb)")
    dxtb_group.add_argument(
        "--dxtb-method",
        type=str,
        default="gfn1",
        choices=["gfn1", "gfn2"],
        help="dxtb method: gfn1 (default) or gfn2",
    )

    # SO3LR-specific options
    so3lr_group = parser.add_argument_group("SO3LR Options")
    so3lr_group.add_argument(
        "--so3lr-model",
        type=str,
        default="so3lr_v1",
        help="SO3LR model: so3lr-s | so3lr-m | so3lr-l | so3lr_v1, or a path to a custom/fine-tuned model (default: so3lr_v1)",
    )
    so3lr_group.add_argument(
        "--lr-cutoff",
        type=float,
        default=1000.0,
        help="SO3LR long-range cutoff in Angstrom (default: 1000 for gas-phase, use 12 for periodic)",
    )
    so3lr_group.add_argument(
        "--charge", type=float, default=0.0, help="Total system charge (default: 0.0)"
    )


def _build_solvation_subparser(subparsers):
    """Build the 'solvation' subparser."""
    solv_parser = subparsers.add_parser(
        "solvation",
        help="Build explicit solvation shells (layer-by-layer)",
        description="Place solvent molecules around a solute in concentric layers, "
        "with optional relaxation after each layer or after all layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AUTOMATIC: grow water shells until 5 Å of padding is reached
  mars solvation solute.xyz --solvent water --padding 5.0

  # AUTOMATIC: grow until at least 30 molecules are placed
  mars solvation solute.xyz --solvent water --n-molecules 30

  # AUTOMATIC: 6 Å padding with layerwise SO3LR relaxation
  mars solvation solute.xyz --solvent water --padding 6.0 --opt-mode layerwise

  # MANUAL: 6 water molecules around a solute, no optimisation
  mars solvation solute.xyz --solvent water --layers 6 --opt-mode none

  # MANUAL: Two-shell water solvation (6 + 12), optimise after all layers
  mars solvation solute.xyz --solvent water --layers 6 12 --opt-mode after-all

  # MANUAL: Three-shell methanol, relax each layer independently
  mars solvation solute.xyz --solvent methanol --layers 4 8 12 --opt-mode layerwise

  # Charged system with SO3LR, tight convergence
  mars solvation solute.xyz --solvent water --padding 5.0 --charge -1 --fmax 0.01

  # Use MACE potential for optimisation
  mars solvation solute.xyz --solvent water --padding 5.0 \\
      --potential mace --opt-mode layerwise

  # Custom placement parameters
  mars solvation solute.xyz --solvent acetonitrile --layers 4 8 \\
      --vdw-scale 0.90 --buffer 0.5 --n-candidates 300 --n-shells 4
        """,
    )

    # Required positional
    solv_parser.add_argument(
        "input",
        type=str,
        help="Solute structure file (.xyz)",
    )

    # Solvation setup
    setup_group = solv_parser.add_argument_group("Solvation Setup")
    setup_group.add_argument(
        "--solvent",
        type=str,
        default="water",
        help='Solvent name (default: water). Use `python -c "import mars.solvents; '
        'print(mars.solvents.list_solvents())"` to see all available solvents, '
        "or pass an alias (e.g. 'hexane', 'DCM', 'EtOH', 'DMSO').",
    )

    # Auto vs manual mode — these three are mutually exclusive convergence criteria
    auto_group = solv_parser.add_argument_group("Automatic mode (grows shells until convergence)")
    auto_group.add_argument(
        "--padding",
        type=float,
        default=None,
        metavar="Å",
        help="Target shell thickness in Å around the solute.  "
        "Shells are added automatically until this padding is reached. "
        "Default when neither --padding, --n-molecules, nor --layers is given: 5.0",
    )
    auto_group.add_argument(
        "--n-molecules",
        type=int,
        default=None,
        metavar="N",
        help="Target total number of solvent molecules.  "
        "Shells are added automatically until this count is reached.",
    )
    auto_group.add_argument(
        "--max-layers",
        type=int,
        default=10,
        help="Safety cap on the number of auto-generated shells (default: 10)",
    )

    manual_group = solv_parser.add_argument_group("Manual mode (explicit layer counts)")
    manual_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Number of solvent molecules per layer.  "
        "Each value adds one concentric shell. "
        "Examples: --layers 6  (one shell)  --layers 6 12  (two shells). "
        "When given, --padding and --n-molecules are ignored.",
    )

    # Optimisation
    opt_group = solv_parser.add_argument_group("Optimisation")
    opt_group.add_argument(
        "--opt-mode",
        type=str,
        default="layerwise",
        choices=["none", "after-all", "layerwise"],
        help="When to run geometry optimisation (default: layerwise):\n"
        "  layerwise  — DEFAULT. Iteratively fill a shell with as many solvents\n"
        "               as fit, relax (solute frozen), refill until saturated,\n"
        "               then grow the next shell until the solute is covered.\n"
        "               Needs a working potential (SO3LR/MACE); slower.\n"
        "  after-all  — fill all shells, then optimise the whole shell once\n"
        "  none       — placement only, no energy minimisation (fast)",
    )
    opt_group.add_argument(
        "--freeze-mode",
        type=str,
        default="solute",
        choices=["solute", "solvent", "none"],
        help="Which atoms to freeze during optimisation "
        "(only used with after-all mode). Default: solute",
    )
    opt_group.add_argument(
        "--method",
        type=str,
        default="FIRE",
        choices=["FIRE", "LBFGS", "GD", "HYBRID"],
        help="Optimisation method (default: FIRE — robust for solvation "
        "relaxation; HYBRID = FIRE coarse + LBFGS fine refinement)",
    )
    opt_group.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion in eV/Å (default: 0.05)",
    )
    opt_group.add_argument(
        "--maxiter",
        type=int,
        default=10000,
        help="Maximum optimisation iterations per layer (default: 10000)",
    )
    opt_group.add_argument(
        "--max-stepsize",
        type=float,
        default=0.15,
        help="Maximum per-atom step in Å — LBFGS step bound, SP trust radius "
        "(not used by FIRE; default: 0.15)",
    )
    opt_group.add_argument(
        "--fire-dt-start",
        type=float,
        default=0.05,
        help="FIRE initial time step (default: 0.05)",
    )
    opt_group.add_argument(
        "--fire-dt-max",
        type=float,
        default=0.1,
        help="FIRE maximum time step (default: 0.1)",
    )
    opt_group.add_argument(
        "--fire-n-min",
        type=int,
        default=2,
        help="FIRE minimum positive-power steps before dt increase (default: 2)",
    )

    # Placement parameters
    place_group = solv_parser.add_argument_group("Placement Parameters")
    place_group.add_argument(
        "--vdw-scale",
        type=float,
        default=0.75,
        help="vdW overlap rejection scale factor (default: 0.75, "
        "lower = tighter packing/more vdW overlap in the pre-relax structure)",
    )
    place_group.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Extra gap in Å beyond vdW contact before placing candidates " "(default: 0.0)",
    )
    place_group.add_argument(
        "--n-candidates",
        type=int,
        default=300,
        help="Fibonacci-sphere candidate points per atom per radial shell " "(default: 300)",
    )
    place_group.add_argument(
        "--n-shells",
        type=int,
        default=3,
        help="Number of radial candidate shells per atom (default: 3)",
    )
    place_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed — layer i uses seed+i (default: 42)",
    )
    place_group.add_argument(
        "--n-orient",
        type=int,
        default=8,
        help="Orientation tries per candidate site in the cover (layerwise/auto) "
        "mode — lets elongated solvents find a tangent fit (default: 8)",
    )
    place_group.add_argument(
        "--refill-cycles",
        type=int,
        default=3,
        help="Max fill→relax→refill cycles per shell before advancing " "(cover mode; default: 3)",
    )

    # Topology-repair options (post-relaxation solvent integrity)
    topo_group = solv_parser.add_argument_group(
        "Topology repair (post-relaxation solvent integrity check)"
    )
    topo_group.add_argument(
        "--no-topology-repair",
        dest="topology_repair",
        action="store_false",
        help="Disable the post-relaxation topology check. By default "
        "(layerwise/auto) solvent molecules that fragment or react during "
        "relaxation are removed, refilled to the target, and re-relaxed until "
        "none are broken; the final structure is also cleaned of broken "
        "molecules in every mode.",
    )
    topo_group.set_defaults(topology_repair=True)
    topo_group.add_argument(
        "--max-repair",
        type=int,
        default=3,
        help="Max remove→refill→relax topology-repair passes (default: 3)",
    )
    topo_group.add_argument(
        "--topo-tolerance",
        type=float,
        default=1.3,
        help="Covalent-bond tolerance for the topology check — a bond counts "
        "as broken past tol×(r_i+r_j) (default: 1.3)",
    )

    # Barostat (spherical moving-wall droplet) mode
    baro_group = solv_parser.add_argument_group(
        "Barostat mode (spherical moving-wall droplet, then swap)"
    )
    baro_group.add_argument(
        "--barostat",
        action="store_true",
        help="Build the solvent shell as a separately compressed spherical "
        "droplet (inner cavity wall + outer 'piston' wall moved inward until a "
        "target reaction force), then swap the solute into the cavity. Requires "
        "--n-molecules N.",
    )
    baro_group.add_argument(
        "--baro-inner-buffer",
        type=float,
        default=2.0,
        help="Cavity clearance (Å) beyond the solute bounding radius (default: 2.0)",
    )
    baro_group.add_argument(
        "--baro-radius",
        type=float,
        default=None,
        help="Initial outer-wall radius in Å (default: auto from --baro-fill)",
    )
    baro_group.add_argument(
        "--baro-fill",
        type=float,
        default=0.35,
        help="Initial droplet fill fraction used to size the outer radius "
        "(lower = looser start; default: 0.35)",
    )
    baro_group.add_argument(
        "--baro-step",
        type=float,
        default=0.5,
        help="Coarse outer-wall compression step in Å per barostat iteration (default: 0.5)",
    )
    baro_group.add_argument(
        "--baro-fine-step",
        type=float,
        default=0.1,
        help="Fine compression step in Å near the target — the wall creeps in by "
        "this amount until the force target is met (default: 0.1)",
    )
    baro_group.add_argument(
        "--baro-k",
        type=float,
        default=5.0,
        help="Wall force constant in eV/Å² (default: 5.0)",
    )
    baro_group.add_argument(
        "--baro-wall-mode",
        type=str,
        default="harmonic",
        choices=["harmonic", "wall"],
        help="Wall form: harmonic (n=2, soft) or wall (n=12, steep) (default: harmonic)",
    )
    baro_group.add_argument(
        "--baro-max-force",
        type=float,
        default=1.0,
        help="Stop compression when the outer-wall reaction force reaches this "
        "(eV/Å) — the target 'pressure' (default: 1.0)",
    )
    baro_group.add_argument(
        "--baro-max-steps",
        type=int,
        default=40,
        help="Maximum barostat compression steps (default: 40)",
    )
    baro_group.add_argument(
        "--baro-md",
        action="store_true",
        help="Drive compression with a short NVT MD (wall in the energy) at each "
        "step before the FIRE settle — lets the solvent flow/pack like a liquid",
    )
    baro_group.add_argument(
        "--baro-md-temp",
        type=float,
        default=300.0,
        help="Temperature (K) for --baro-md compression (default: 300)",
    )
    baro_group.add_argument(
        "--baro-md-time",
        type=float,
        default=0.3,
        help="MD time per compression step in ps for --baro-md (default: 0.3)",
    )
    baro_group.add_argument(
        "--baro-md-dt",
        type=float,
        default=0.5,
        help="MD timestep in fs for --baro-md (default: 0.5)",
    )

    # Potential options (reused pattern from optimize)
    pot_group = solv_parser.add_argument_group("Potential Options (used when opt-mode != none)")
    pot_group.add_argument(
        "--potential",
        type=str,
        default="so3lr",
        help="Potential: so3lr, mace, dxtb (default: so3lr)",
    )
    pot_group.add_argument(
        "--charge",
        type=float,
        default=0.0,
        help="Total system charge (default: 0.0)",
    )
    pot_group.add_argument(
        "--so3lr-model",
        type=str,
        default="so3lr_v1",
        help="SO3LR model: so3lr-s | so3lr-m | so3lr-l | so3lr_v1, or a path to a custom/fine-tuned model (default: so3lr_v1)",
    )
    pot_group.add_argument(
        "--lr-cutoff",
        type=float,
        default=1000.0,
        help="SO3LR long-range cutoff in Å (default: 1000)",
    )
    pot_group.add_argument(
        "--mace-foundation",
        type=str,
        default="off",
        choices=["mp", "off", "anicc", "omol"],
        help="MACE foundation family: off (default), mp, anicc, omol",
    )
    pot_group.add_argument(
        "--mace-model",
        type=str,
        default="small",
        help="MACE variant name: small (default), medium, large, ... "
        "(omol only has extra_large); may also be a local path or URL",
    )
    pot_group.add_argument(
        "--mace-cache-dir",
        type=str,
        default=None,
        help="Directory for converted JAX weights (default: ~/.cache/mars/mace_jax)",
    )
    pot_group.add_argument(
        "--spin",
        type=float,
        default=1.0,
        help="Spin multiplicity for charge/spin-aware MACE models (omol); default 1",
    )
    pot_group.add_argument(
        "--dxtb-method",
        type=str,
        default="gfn1",
        choices=["gfn1", "gfn2"],
        help="dxtb method (default: gfn1)",
    )

    # Output
    out_group = solv_parser.add_argument_group("Output")
    out_group.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file. Single-conformer/replica: defaults to solvated.xyz. "
        "Multi-conformer/replica: when given, the multi-frame ensemble is "
        "written here; otherwise <output-prefix>_all.xyz is used.",
    )

    # Multi-conformer / Replica
    multi_group = solv_parser.add_argument_group("Multi-conformer / Replica Options")
    multi_group.add_argument(
        "--conformer",
        type=int,
        default=None,
        metavar="N",
        help="For multi-frame XYZ: 0-based index of the conformer to solvate "
        "(default: 0). Also used as reference when --solute-swap is given",
    )
    multi_group.add_argument(
        "--all-conformers",
        action="store_true",
        help="Solvate each conformer in a multi-frame XYZ independently "
        "(separate solvation per frame)",
    )
    multi_group.add_argument(
        "--solute-swap",
        action="store_true",
        help="Solvate the reference conformer, then transplant the solvent "
        "shell onto every other conformer and re-optimise. "
        "Use --conformer to select the reference frame (default: 0)",
    )
    multi_group.add_argument(
        "--no-align",
        action="store_true",
        help="Disable RMSD alignment of conformers to the reference before "
        "transplanting the solvent shell (default: align is ON). "
        "⚠️  WARNING: Kabsch alignment can introduce errors for reaction "
        "trajectories with large conformational changes, symmetries, or "
        "non-representative references. Use --no-align or --all-conformers "
        "for more accurate solvation in such cases.",
    )
    multi_group.add_argument(
        "--relax-solute",
        action="store_true",
        help="Also optimise solute atoms during the constrained optimisation "
        "step (default: freeze solute, relax only solvent)",
    )
    multi_group.add_argument(
        "--final-relax",
        nargs="?",
        type=float,
        const=-1.0,  # sentinel: flag given without a value → use the process --fmax
        default=None,
        metavar="FMAX",
        help="Run a final unconstrained relaxation (solute + solvent free) "
        "after the constrained optimisation (default: OFF). Pass the flag alone "
        "to use the same --fmax as the rest of the process, or give a value "
        "(e.g. --final-relax 0.01) to use a tighter force threshold for the "
        "final relax only.",
    )
    multi_group.add_argument(
        "--n-solvations",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent solvation replicas with different "
        "random seeds (default: 1). Seeds: seed, seed+1, ..., seed+(N-1)",
    )
    multi_group.add_argument(
        "--output-prefix",
        type=str,
        default="solvated",
        help="Prefix for per-conformer/replica output files (default: solvated). "
        "Files: <prefix>_confI_repJ.xyz",
    )
    multi_group.add_argument(
        "--not-parallel",
        action="store_true",
        default=False,
        help="Disable vmap parallelism for batch optimisation "
        "(default: parallel when batch > 1)",
    )
    multi_group.add_argument(
        "--no-multistage",
        action="store_true",
        default=False,
        help="Disable two-stage (coarse → fine) optimisation. "
        "Use a single optimisation pass instead (default: multistage ON)",
    )

    # Computational
    comp_group = solv_parser.add_argument_group("Computational Options")
    comp_group.add_argument(
        "--float64",
        action="store_true",
        help="Use 64-bit floating point (default: 32-bit)",
    )
    comp_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution",
    )

    # Logging
    log_group = solv_parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: console only)",
    )
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return solv_parser


def _build_viewer_subparser(subparsers):
    """Build the 'viewer' subparser (interactive 3D / IR / NCI visualization)."""
    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Interactive 3D viewer for molecules, IR results, and the NCI wall",
        description="Visualise structures and results: a plain 3D molecule, an "
        "interactive harmonic-IR explorer, or the spherical confinement wall "
        "used by --nci. Requires matplotlib (pip install matplotlib).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plain 3D molecule viewer (rotate/zoom with the mouse)
  mars viewer molecule.xyz

  # Interactive IR explorer — auto-discover ir results in the current directory
  mars ir molecule.xyz --mode-xyz --save-structure --plot
  mars viewer --ir

  # IR explorer pointing at a results directory or explicit files
  mars viewer --ir --dir results/
  mars viewer --ir --modes normal_modes.xyz --spectrum ir_spectrum.dat

  # Preview the NCI confinement wall and check the clearance/buffer
  mars viewer complex.xyz --nci
  mars viewer complex.xyz --nci --nci-buffer 5.0

  # Render headlessly to a PNG instead of opening a window (no display needed)
  mars viewer molecule.xyz --save view.png
  mars viewer --ir --save ir_view.png

Controls (IR explorer):
  ◄ Prev / Next ►   step through vibrational modes
  click a peak      select the nearest mode
  ← / →             previous / next mode (keyboard)
  space             toggle the vibration animation
  Arrow / Amplitude sliders adjust arrow length and animation amplitude
        """,
    )

    viewer_parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input structure file (XYZ). Required for plain and --nci modes; "
        "ignored in --ir mode (IR data is loaded from result files).",
    )

    mode_group = viewer_parser.add_argument_group("Viewer Mode")
    mode_group.add_argument(
        "--ir",
        action="store_true",
        help="Interactive IR explorer: 3D molecule with per-mode displacement "
        "arrows + animation, an info panel, and a clickable IR spectrum.",
    )
    mode_group.add_argument(
        "--nci",
        action="store_true",
        help="Show the molecule inside the spherical confinement wall used by "
        "`mars ... --nci`, to judge whether --nci-buffer is large enough.",
    )
    mode_group.add_argument(
        "--nci-buffer",
        type=float,
        default=3.0,
        metavar="Å",
        help="Buffer distance (Å) between the outermost atom and the "
        "confinement wall (default: 3.0). Only used with --nci.",
    )

    ir_group = viewer_parser.add_argument_group("IR File Selection (--ir mode)")
    ir_group.add_argument(
        "--dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to search for IR result files (default: current "
        "directory). Explicit --modes/--spectrum/--mode-analysis override this.",
    )
    ir_group.add_argument(
        "--modes",
        type=str,
        default=None,
        metavar="FILE",
        help="Normal-modes XYZ file with eigenvectors (default: normal_modes.xyz)",
    )
    ir_group.add_argument(
        "--spectrum",
        type=str,
        default=None,
        metavar="FILE",
        help="IR spectrum data file with intensities (default: ir_spectrum.dat)",
    )
    ir_group.add_argument(
        "--mode-analysis",
        type=str,
        default=None,
        metavar="FILE",
        help="Mode-classification file for colour/labels (default: mode_analysis.txt)",
    )

    display_group = viewer_parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--atom-scale",
        type=float,
        default=1.0,
        help="Atom marker-size multiplier (default: 1.0; larger = bigger atoms)",
    )
    display_group.add_argument(
        "--bond-width",
        type=float,
        default=2.5,
        help="Bond line width in points (default: 2.5)",
    )
    display_group.add_argument(
        "--grid",
        action="store_true",
        help="Show the 3D grid, axes and pane backgrounds (default: hidden for "
        "a clean molecule-only view; also toggleable live in the --ir window)",
    )
    display_group.add_argument(
        "--arrow-scale",
        type=float,
        default=1.0,
        help="Initial displacement-arrow length multiplier (default: 1.0)",
    )
    display_group.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        metavar="Å",
        help="Initial vibration-animation amplitude in Å (default: 0.5)",
    )
    display_group.add_argument(
        "--broadening",
        type=float,
        default=10.0,
        help="Lorentzian broadening width in cm⁻¹ for the spectrum " "envelope (default: 10.0)",
    )
    display_group.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=[400, 4000],
        metavar=("MIN", "MAX"),
        help="Frequency range for the spectrum panel in cm⁻¹ " "(default: 400 4000)",
    )
    display_group.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Render the view to this PNG (headless, no window) instead of "
        "opening an interactive window.",
    )

    log_group = viewer_parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-file", type=str, default=None, help="Log file path (default: console only)"
    )
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    log_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode: verbose logging"
    )

    return viewer_parser
