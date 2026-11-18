#!/usr/bin/env python3
"""
Benchmark: average completion time vs number of processes / Pis.

Launch ONE mpirun using every core in the cluster; the script then times
the same job on growing subsets of those processes (using sub-communicators):

    mpirun -np 12 --hostfile hosts python3 mpi_benchmark.py 50000000 10
    mpirun -np 12 --hostfile hosts python3 mpi_benchmark.py 100000000 10

Arguments (all optional):
    size       sieve: N, the upper limit.  pi: number of random samples.
    repeats    timed repetitions per process count (default 10)
    counts     comma-separated process counts, e.g. 1,2,4,8,12
               (default: powers of two up to -np, plus -np itself)

With 4 slots per Pi and the head node listed first in the hostfile,
ranks 0-3 live on the head, 4-7 on node1, 8-11 on node2, so
4 processes = 1 Pi, 8 = 2 Pis, 12 = 3 Pis. The "nodes" column in the
output shows the real placement.

The total amount of work is the SAME for every process count (strong
scaling), so the time should fall as processes are added.
"""
import math
import statistics
import sys

import numpy as np
from mpi4py import MPI

from mpi_sieve import block_range, simple_sieve, block_sieve


# ---------------------------------------------------------------- workloads
def run_sieve(comm, N, trial):
    """Count primes up to N using the processes in `comm`."""
    rank, size = comm.Get_rank(), comm.Get_size()
    comm.Barrier()
    t_start = MPI_Wtime()
    base_primes = simple_sieve(math.isqrt(N))
    low, high = block_range(N - 1, size, rank, 2)
    primes, local_count, local_max = block_sieve(low, high, base_primes)
    t_local = MPI.Wtime() - t_start
    return comm.reduce(count, op=MPI.SUM, root=0)


# --------------------------------------------------------------------- main
def default_counts(world):
    counts, k = [], 1
    while k < world:
        counts.append(k)
        k *= 2
    counts.append(world)
    return counts


def main():
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()

    size_arg = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if len(sys.argv) > 3:
        counts = [int(c) for c in sys.argv[4].split(",")]
    else:
        counts = default_counts(world)
    counts = [k for k in counts if 1 <= k <= world]


    names = comm.allgather(MPI.Get_processor_name())
    results = []          # (k, nodes, mean, stdev, best)
    reference = None      # first sieve answer, used to check later ones

    for k in counts:
        color = 0 if rank < k else MPI.UNDEFINED
        sub = comm.Split(color, rank)
        times = []

        # trial 0 is a warm-up and is thrown away
        for trial in range(reps + 1):
            comm.Barrier()                      # everyone starts together
            if sub != MPI.COMM_NULL:
                t0 = MPI.Wtime()
                result = run_sieve(sub, size_arg, trial)
                sub.Barrier()                   # wait for the slowest process
                t = MPI.Wtime() - t0
                if rank == 0 and trial > 0:
                    times.append(t)
                if rank == 0:
                    if reference is None:
                        reference = result
                    elif result != reference:
                        print(f"WARNING: wrong count with {k} processes: "
                              f"{result} != {reference}")

        if sub != MPI.COMM_NULL:
            sub.Free()

        if rank == 0:
            mean = statistics.mean(times)
            sd = statistics.stdev(times) if len(times) > 1 else 0.0
            results.append((k, len(set(names[:k])), mean, sd, min(times)))
            print(f"  finished {k} process(es): mean {mean:.3f}s", flush=True)

    if rank == 0:
        base_k, base_t = results[0][0], results[0][2]
        print(f"\nWorkload: {run_sieve}, size {size_arg}, "
              f"{reps} timed runs per row (+1 warm-up)")
        print(f"Primes found (checked identical for every row): {reference}")
        print(f"\n{'procs':>5} {'nodes':>5} {'mean(s)':>9} {'std(s)':>8} "
              f"{'best(s)':>9} {'speedup':>8} {'efficiency':>10}")
        rows = []
        for k, nodes, mean, sd, best in results:
            speedup = base_t / mean
            eff = speedup * base_k / k
            rows.append([k, nodes, mean, sd, best, speedup, eff])
            print(f"{k:>5} {nodes:>5} {mean:>9.3f} {sd:>8.3f} "
                  f"{best:>9.3f} {speedup:>7.2f}x {eff:>9.0%}")

if __name__ == "__main__":
    main()
