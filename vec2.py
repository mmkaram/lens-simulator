import torch


class Vec2:
    def __init__(self, x, y):
        # Store as a tensor for batch operations later
        self.data = torch.tensor([x, y], dtype=torch.float64)

    @property
    def x(self):
        return self.data[0].item()

    @property
    def y(self):
        return self.data[1].item()

    def dot(self, other):
        return torch.dot(self.data, other.data).item()

    def length(self):
        return torch.norm(self.data).item()

    def normalize(self):
        normalized = self.data / torch.norm(self.data)
        return Vec2(normalized[0].item(), normalized[1].item())

    def scale(self, scalar):
        scaled = self.data * scalar
        return Vec2(scaled[0].item(), scaled[1].item())

    def add(self, other):
        result = self.data + other.data
        return Vec2(result[0].item(), result[1].item())

    def sub(self, other):
        result = self.data - other.data
        return Vec2(result[0].item(), result[1].item())
