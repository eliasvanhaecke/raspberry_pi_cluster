#!/usr/bin/env python3
"""
Sieve baseline for the sieve: no MPI, one process, one core.

Usage:
    taskset -c 0 python3 sieve_baseline.py 50000000 10

Arguments (optional): N (default 20,000,000) and number of timed repeats (default 10).
"""
import statistics
import sys
import time

import numpy as np


def count_primes(N):
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return int(sieve.sum())


def main():
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    times, count = [], None
    for trial in range(reps + 1):
        t0 = time.perf_counter()
        count = count_primes(N)
        t = time.perf_counter() - t0
        if trial > 0:
            times.append(t)

    median = statistics.median(times)
    mean = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"Serial sieve, N = {N}, {reps} timed runs (+1 warm-up)")
    print(f"Primes found: {count}")
    print(f"mean {mean:.3f}s   median {median:.3f}   std {sd:.3f}s   best {min(times):.3f}s")


if __name__ == "__main__":
    main()
