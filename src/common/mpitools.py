from mpi4py import MPI
import numpy as np
import torch

def get_comm():
    return MPI.COMM_WORLD

def rank_world(comm=None):
    comm = comm or get_comm()
    return comm.Get_rank(), comm.Get_size()

def allreduce_tensor_sum_(comm, t: torch.Tensor):
    """
    In-place allreduce SUM on CPU tensors. Converts to numpy buffer.
    """
    assert not t.is_cuda, "This skeleton assumes CPU tensors for mpi4py simplicity."
    buf = t.detach().contiguous().numpy()
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)
    t.copy_(torch.from_numpy(out))

def broadcast_model_params(comm, model, root=0):
    """
    Broadcast model parameters from root to all ranks.
    """
    for p in model.parameters():
        t = p.data
        if comm.Get_rank() == root:
            buf = t.detach().contiguous().numpy()
        else:
            buf = np.empty(t.shape, dtype=np.float32)
        comm.Bcast(buf, root=root)
        if comm.Get_rank() != root:
            p.data.copy_(torch.from_numpy(buf))

def sync_gradients_avg(comm, model):
    """
    Allreduce gradients and average across ranks.
    """
    world = comm.Get_size()
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        allreduce_tensor_sum_(comm, g)
        g /= world

def split_by_color(comm, color: int, key: int = 0):
    """
    Create a sub-communicator.
    """
    return comm.Split(color=color, key=key)

def send_tensor(comm, t: torch.Tensor, dest: int, tag: int):
    assert not t.is_cuda
    buf = t.detach().contiguous().numpy()
    comm.Send(buf, dest=dest, tag=tag)

def recv_tensor(comm, shape, dtype, src: int, tag: int) -> torch.Tensor:
    buf = np.empty(shape, dtype=dtype)
    comm.Recv(buf, source=src, tag=tag)
    return torch.from_numpy(buf)
