# hello.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Rank {comm.Get_rank()} of {comm.Get_size()} on {MPI.Get_processor_name()}")