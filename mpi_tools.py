from mpi4py import MPI
import numpy as np


def num_procs():
    """
    :return: Number of processes
    """
    return MPI.COMM_WORLD.Get_size()


def all_reduce(*args, **kwargs):
    """
    MPI.ALLreduce reduces the values and distribute the results to all the processes, the reduce operation is MPI.SUM
    """
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_op(x, op):
    """
    Executes the all_reduce function, and stores the value in buff, checks if value is scalar or not
    """
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    all_reduce(x, buff, op=op)
    return buff[0] if scalar else buff


def broadcast(x, root=0):
    """
    broadcast the values of x to the root node
    """
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_sum(x):
    """
    Executes the MPI.SUM operation
    """
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    return mpi_sum(x) / num_procs()


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)
