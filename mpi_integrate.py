#!/usr/bin/env python3
"""
Parallel numerical integration using mpi4py (composite Simpson's rule).

Usage:
    mpirun -np 12 --hostfile hosts python3 mpi_integrate.py [integrand] [n]

    integrand  "pi" (default), "sin" or "gauss"   (see INTEGRANDS below)
    n          number of Simpson sub-intervals (default 100_000_000)

Approach:
  1. The interval [a, b] is cut into n equal sub-intervals of width h.
  2. The sub-intervals are split into contiguous blocks, one per rank
     (same block_range() split as mpi_sieve.py).
  3. Each rank applies Simpson's rule to every sub-interval in its block:
         h/6 * ( f(x_i) + 4 f(x_i + h/2) + f(x_i+1) )
     Blocks are completely independent, so NO communication is needed
     during the computation.
  4. The partial sums are combined with a single reduce on rank 0.

Why this parallelises well: the work is embarrassingly parallel (the only
communication is one tiny reduce), and the work per rank is identical, so
the load is balanced. Each rank evaluates its points in chunks so memory
use stays small on the 4 GB / 8 GB Pis.
"""
import math
import sys

import numpy as np
from mpi4py import MPI

from mpi_sieve import block_range

# name: (function on numpy arrays, a, b, exact value)
INTEGRANDS = {
    # integral of 4/(1+x^2) over [0,1] is pi
    "pi": (lambda x: 4.0 / (1.0 + x * x), 0.0, 1.0, math.pi),
    # integral of sin(x) over [0,pi] is 2
    "sin": (np.sin, 0.0, math.pi, 2.0),
    # integral of exp(-x^2) over [-6,6] is sqrt(pi) * erf(6)
    "gauss": (lambda x: np.exp(-x * x), -6.0, 6.0,
              math.sqrt(math.pi) * math.erf(6.0)),
    #"crazy": (lambda x: np.exp(np.sin(x * x) * np.cos(x) * np.cos(x))),
    # crazy function to ensure long compute times
}

CHUNK = 1_000_000  # points evaluated per numpy call


def integrate_block(f, a: float, h: float, lo: int, hi: int):
    """Simpson's rule over sub-intervals lo..hi-1 (indices, hi exclusive).

    Sub-interval i is [a + i*h, a + (i+1)*h]. Returns the partial integral.
    """
    total = 0.0
    start = lo
    while start < hi:
        end = min(start + CHUNK, hi)
        # x_start .. x_end  (end - start + 1 points, shared endpoints
        # between consecutive sub-intervals are evaluated only once)
        x = a + h * np.arange(start, end + 1, dtype=np.float64)
        fx = f(x)
        mid = f(x[:-1] + 0.5 * h)
        total += float(fx[:-1].sum() + fx[1:].sum() + 4.0 * mid.sum())
        start = end
    return total * h / 6.0


def parallel_integrate(comm, name: str, n: int):
    """Integrate INTEGRANDS[name] with n sub-intervals over `comm`.

    Returns (value, lo, hi) on every rank, where value is only meaningful
    (non-None) on rank 0. Used by both main() and mpi_benchmark.py.
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    f, a, b, _ = INTEGRANDS[name]
    h = (b - a) / n

    # this rank's block of sub-intervals 0..n-1
    lo, hi = block_range(n, size, rank, 0)
    hi += 1  # block_range returns an inclusive end; we want exclusive

    local = integrate_block(f, a, h, lo, hi)
    value = comm.reduce(local, op=MPI.SUM, root=0)
    return value, lo, hi


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    name = sys.argv[1] if len(sys.argv) > 1 else "pi"
    n = int(float(sys.argv[2])) if len(sys.argv) > 2 else 100_000_000

    if name not in INTEGRANDS or n < 1:
        if rank == 0:
            print(f"integrand must be one of {list(INTEGRANDS)} and n >= 1")
        return

    comm.Barrier()
    t_start = MPI.Wtime()

    value, lo, hi = parallel_integrate(comm, name, n)

    t_local = MPI.Wtime() - t_start
    slowest = comm.reduce(t_local, op=MPI.MAX, root=0)

    # Gather per-rank stats so output is printed in one place, in rank order
    stats = (rank, MPI.Get_processor_name(), lo, hi, t_local)
    all_stats = comm.gather(stats, root=0)

    if rank == 0:
        print(f"\n{'rank':>4}  {'node':<16} {'sub-intervals':<28} {'time':>8}")
        for r, node, l, h_, t in sorted(all_stats):
            block = f"[{l}, {h_})"
            print(f"{r:>4}  {node:<16} {block:<28} {t:>7.3f}s")
        exact = INTEGRANDS[name][3]
        print(f"\nIntegrand: {name}, n = {n}")
        print(f"Result: {value:.15f}")
        print(f"Exact:  {exact:.15f}")
        print(f"Abs error: {abs(value - exact):.3e}")
        print(f"Processes: {size}, slowest rank time: {slowest:.3f}s")


if __name__ == "__main__":
    main()