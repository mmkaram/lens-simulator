import torch


class RayBatch:
    def __init__(self, pos: torch.Tensor, dir: torch.Tensor):
        assert pos.shape == dir.shape
        self.pos = pos
        self.dir = dir
        self.normalize_dirs()

    def normalize_dirs(self):
        self.dir = self.dir / (torch.norm(self.dir, dim=-1, keepdim=True) + 1e-12)

    def to(self, device):
        self.pos = self.pos.to(device)
        self.dir = self.dir.to(device)
        return self
