import torch
from torch.utils.data import Sampler

class ShardSampler(Sampler):
    """
    Each rank gets indices rank, rank+world, rank+2*world, ...
    """
    def __init__(self, dataset_len: int, rank: int, world: int, shuffle: bool = True, seed: int = 0):
        self.n = dataset_len
        self.rank = rank
        self.world = world
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        idx = torch.arange(self.n)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            idx = idx[torch.randperm(self.n, generator=g)]
        shard = idx[self.rank::self.world].tolist()
        return iter(shard)

    def __len__(self):
        # approximate shard length
        return (self.n + self.world - 1) // self.world
