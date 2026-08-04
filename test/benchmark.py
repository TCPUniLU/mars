"""
Performance benchmarks for JAX implementation.

Compares JAX vs NumPy performance for key operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mars.rmsd import rmsd_cv_jax, kabsch_jax, pairwise_rmsd_matrix
from mars.mtd import create_mtd_state, add_hill, compute_mtd_bias
from mars.utils import enable_float64, get_device_info

# Enable 64-bit precision
enable_float64()


def benchmark_rmsd():
    """Benchmark RMSD computation."""
    print("\n" + "=" * 70)
    print("Benchmark: RMSD Computation")
    print("=" * 70)

    n_atoms = 100
    n_iterations = 1000

    # Generate random structures
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    pos1_jax = jax.random.normal(key1, (n_atoms, 3))
    pos2_jax = jax.random.normal(key2, (n_atoms, 3))

    pos1_np = np.array(pos1_jax)
    pos2_np = np.array(pos2_jax)

    # JAX version (with JIT warmup)
    print(f"Testing with {n_atoms} atoms, {n_iterations} iterations")

    # Warmup
    _ = rmsd_cv_jax(pos1_jax, pos2_jax)

    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        rmsd_jax = rmsd_cv_jax(pos1_jax, pos2_jax)
    jax.block_until_ready(rmsd_jax)  # Wait for GPU
    time_jax = time.time() - start

    # NumPy version (if available)
    try:
        from mars.utils import rmsd as rmsd_np

        start = time.time()
        for _ in range(n_iterations):
            rmsd_numpy = rmsd_np(pos1_np, pos2_np)
        time_numpy = time.time() - start

        print(f"\nNumPy time: {time_numpy:.4f}s ({time_numpy/n_iterations*1000:.4f} ms/iter)")
        print(f"JAX time:   {time_jax:.4f}s ({time_jax/n_iterations*1000:.4f} ms/iter)")
        print(f"Speedup:    {time_numpy/time_jax:.2f}x")
        print(f"RMSD difference: {abs(rmsd_numpy - float(rmsd_jax)):.2e}")

    except ImportError:
        print(f"JAX time: {time_jax:.4f}s ({time_jax/n_iterations*1000:.4f} ms/iter)")
        print("(NumPy version not available for comparison)")


def benchmark_kabsch():
    """Benchmark Kabsch alignment."""
    print("\n" + "=" * 70)
    print("Benchmark: Kabsch Alignment")
    print("=" * 70)

    n_atoms = 100
    n_iterations = 1000

    key = jax.random.PRNGKey(42)
    P_jax = jax.random.normal(key, (n_atoms, 3))
    Q_jax = jax.random.normal(jax.random.PRNGKey(43), (n_atoms, 3))

    # Center
    P_jax = P_jax - P_jax.mean(0)
    Q_jax = Q_jax - Q_jax.mean(0)

    # Warmup
    _ = kabsch_jax(P_jax, Q_jax)

    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        R = kabsch_jax(P_jax, Q_jax)
    jax.block_until_ready(R)
    time_jax = time.time() - start

    print(f"JAX Kabsch: {time_jax:.4f}s ({time_jax/n_iterations*1000:.4f} ms/iter)")


def benchmark_mtd_bias():
    """Benchmark MTD bias computation."""
    print("\n" + "=" * 70)
    print("Benchmark: MTD Bias Computation")
    print("=" * 70)

    n_atoms = 100
    n_hills_list = [10, 100, 500, 1000]

    positions = jax.random.normal(jax.random.PRNGKey(42), (n_atoms, 3))

    print(f"Testing with {n_atoms} atoms")
    print(f"{'Hills':<10} {'Time (ms)':<15} {'Time/hill (ms)':<20}")
    print("-" * 45)

    for n_hills in n_hills_list:
        # Create state with hills
        state = create_mtd_state(n_atoms=n_atoms, max_hills=n_hills + 10, kpush=0.8, alpha=0.5)

        for i in range(n_hills):
            hill_pos = jax.random.normal(jax.random.PRNGKey(i), (n_atoms, 3))
            state = add_hill(state, hill_pos)

        # Warmup
        _ = compute_mtd_bias(positions, state)

        # Benchmark
        n_iterations = max(10, 1000 // n_hills)  # Fewer iterations for more hills

        start = time.time()
        for _ in range(n_iterations):
            bias = compute_mtd_bias(positions, state)
        jax.block_until_ready(bias)
        elapsed = time.time() - start

        time_per_call = elapsed / n_iterations * 1000  # ms
        time_per_hill = time_per_call / n_hills

        print(f"{n_hills:<10} {time_per_call:<15.4f} {time_per_hill:<20.6f}")


def benchmark_pairwise_rmsd():
    """Benchmark pairwise RMSD matrix."""
    print("\n" + "=" * 70)
    print("Benchmark: Pairwise RMSD Matrix")
    print("=" * 70)

    n_atoms = 50
    n_structures_list = [10, 20, 50]

    print(f"Testing with {n_atoms} atoms per structure")
    print(f"{'Structures':<15} {'Time (s)':<15} {'Time/pair (ms)':<20}")
    print("-" * 50)

    for n_structures in n_structures_list:
        # Generate structures
        key = jax.random.PRNGKey(42)
        structures = jax.random.normal(key, (n_structures, n_atoms, 3))

        # Warmup
        _ = pairwise_rmsd_matrix(structures)

        # Benchmark
        start = time.time()
        rmsd_mat = pairwise_rmsd_matrix(structures)
        jax.block_until_ready(rmsd_mat)
        elapsed = time.time() - start

        n_pairs = n_structures * (n_structures - 1) // 2
        time_per_pair = elapsed / n_pairs * 1000  # ms

        print(f"{n_structures:<15} {elapsed:<15.4f} {time_per_pair:<20.4f}")


def benchmark_gradient_computation():
    """Benchmark gradient computation via autodiff."""
    print("\n" + "=" * 70)
    print("Benchmark: Gradient Computation (Autodiff)")
    print("=" * 70)

    n_atoms_list = [10, 50, 100, 200]

    def energy_fn(positions):
        """Simple harmonic potential."""
        return jnp.sum(positions**2) * 0.1

    print(f"{'Atoms':<10} {'Forward (ms)':<20} {'Forward+Grad (ms)':<25}")
    print("-" * 55)

    for n_atoms in n_atoms_list:
        positions = jax.random.normal(jax.random.PRNGKey(42), (n_atoms, 3))

        # Warmup
        _ = energy_fn(positions)
        _ = jax.grad(energy_fn)(positions)

        n_iterations = 1000

        # Forward pass only
        start = time.time()
        for _ in range(n_iterations):
            E = energy_fn(positions)
        jax.block_until_ready(E)
        time_forward = (time.time() - start) / n_iterations * 1000

        # Forward + gradient
        grad_fn = jax.grad(energy_fn)
        start = time.time()
        for _ in range(n_iterations):
            grad = grad_fn(positions)
        jax.block_until_ready(grad)
        time_grad = (time.time() - start) / n_iterations * 1000

        print(f"{n_atoms:<10} {time_forward:<20.4f} {time_grad:<25.4f}")


def benchmark_jit_compilation():
    """Benchmark JIT compilation speedup."""
    print("\n" + "=" * 70)
    print("Benchmark: JIT Compilation Speedup")
    print("=" * 70)

    n_atoms = 100

    @jax.jit
    def energy_jit(positions):
        return jnp.sum(positions**2) * 0.1

    def energy_no_jit(positions):
        return jnp.sum(positions**2) * 0.1

    positions = jax.random.normal(jax.random.PRNGKey(42), (n_atoms, 3))

    # Warmup JIT
    _ = energy_jit(positions)

    n_iterations = 10000

    # JIT version
    start = time.time()
    for _ in range(n_iterations):
        E = energy_jit(positions)
    jax.block_until_ready(E)
    time_jit = time.time() - start

    # No JIT
    start = time.time()
    for _ in range(n_iterations):
        E = energy_no_jit(positions)
    jax.block_until_ready(E)
    time_no_jit = time.time() - start

    print(f"With JIT:    {time_jit:.4f}s ({time_jit/n_iterations*1e6:.4f} µs/iter)")
    print(f"Without JIT: {time_no_jit:.4f}s ({time_no_jit/n_iterations*1e6:.4f} µs/iter)")
    print(f"Speedup:     {time_no_jit/time_jit:.2f}x")


def print_device_info():
    """Print device information."""
    print("\n" + "=" * 70)
    print("Device Information")
    print("=" * 70)

    info = get_device_info()

    print(f"Available devices: {info['device_count']}")
    for i, device in enumerate(info["devices"]):
        print(f"  Device {i}: {device.device_kind} - {device.platform}")

    print(f"\nDefault device: {info['default_device'].platform}")
    print(f"Has GPU: {info['has_gpu']}")

    if info["has_gpu"]:
        print("\n✓ GPU acceleration available!")
    else:
        print("\n⚠ Running on CPU only")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("MARS Performance Benchmarks")
    print("=" * 70)

    print_device_info()

    benchmarks = [
        benchmark_rmsd,
        benchmark_kabsch,
        benchmark_mtd_bias,
        benchmark_pairwise_rmsd,
        benchmark_gradient_computation,
        benchmark_jit_compilation,
    ]

    for benchmark in benchmarks:
        try:
            benchmark()
        except Exception as e:
            print(f"\n✗ {benchmark.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Benchmarks Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
