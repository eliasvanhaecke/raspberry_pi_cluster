#!/usr/bin/env python3
"""
Parallel Sieve of Eratosthenes using mpi4py (segmented sieve).

Usage:
    mpirun -np 8 --hostfile hosts python3 mpi_sieve.py 100000000

Approach:
  1. Every rank sieves 2..sqrt(N) itself to get the "base primes".
     This is tiny (sqrt(N) numbers), so repeating it is cheaper than
     communicating it.
  2. The range 2..N is split into contiguous blocks, one per rank.
  3. Each rank crosses off multiples of the base primes inside its own block.
  4. Results are combined with reduce (count, largest prime).
"""
import math
import sys

import numpy as np
from mpi4py import MPI


def simple_sieve(limit):
    """Ordinary sieve up to `limit`; returns an array of primes."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def block_range(n_total, size, rank, offset):
    """Split n_total numbers (starting at `offset`) evenly; return inclusive [low, high]."""
    chunk, rem = divmod(n_total, size)
    low = offset + rank * chunk + min(rank, rem)
    high = low + chunk + (1 if rank < rem else 0) - 1
    return low, high


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    if N < 2:
        if rank == 0:
            print("N must be at least 2")
        return

    comm.Barrier()
    t_start = MPI.Wtime()

    # Step 1: base primes, computed redundantly on every rank
    base_primes = simple_sieve(math.isqrt(N))

    # Step 2: this rank's block of [2, N]
    low, high = block_range(N - 1, size, rank, 2)

    # Step 3: sieve the block
    local_count = 0
    local_max = 0
    if high >= low:
        segment = np.ones(high - low + 1, dtype=bool)
        for p in base_primes:
            p = int(p)
            if p * p > high:
                break
            # first multiple of p in the block, but never p itself
            start = max(p * p, ((low + p - 1) // p) * p)
            segment[start - low::p] = False
        primes = np.nonzero(segment)[0] + low
        local_count = int(primes.size)
        if local_count:
            local_max = int(primes[-1])

    t_local = MPI.Wtime() - t_start

    # Step 4: combine results on rank 0
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    largest = comm.reduce(local_max, op=MPI.MAX, root=0)
    slowest = comm.reduce(t_local, op=MPI.MAX, root=0)

    print(f"[rank {rank} on {MPI.Get_processor_name()}] "
          f"block {low}..{high}, found {local_count} primes in {t_local:.3f}s",
          flush=True)

    if rank == 0:
        print(f"\nPrimes up to {N}: {total_count}")
        print(f"Largest prime found: {largest}")
        print(f"Processes: {size}, slowest rank time: {slowest:.3f}s")


if __name__ == "__main__":
    main()
