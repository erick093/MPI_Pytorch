from mpi4py import MPI
import numpy as np


def num_processes():
    """
    Return the total number of processes
    """
    return MPI.COMM_WORLD.Get_size()


def mpi_all_reduce(*args, **kwargs):
    """
    MPI.ALLreduce reduces the values and distribute the results to all the processes, the reduce operation is MPI.SUM
    """
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_sum(x, op):
    """
    Executes the all_reduce function, and stores the value in buff, checks if value is scalar or not
    """
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)  # create the buffer for All Reduce
    mpi_all_reduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_avg_grads(model):
    """ Average the gradients across all MPI processes. """
    if num_processes() == 1:  # if number of processes is 1, then return none
        return None
    for p in model.parameters():
        p_grad_numpy = p.grad.numpy()  # convert tensor to numpy array
        avg_p_grad = mpi_sum(p.grad, MPI.SUM) / num_processes()  # sum all the gradients of every process and average them
        p_grad_numpy[:] = avg_p_grad[:]


def mpi_broadcast(x, root=0):
    """
    Broadcast the values of x to the root node
    """
    MPI.COMM_WORLD.Bcast(x, root=root)


def sync_params(model):
    """ Synchronize the parameters of a model across all MPI processes. """
    if num_processes() == 1:
        return None
    for p in model.parameters():
        p_numpy = p.data.numpy()
        mpi_broadcast(p_numpy)
