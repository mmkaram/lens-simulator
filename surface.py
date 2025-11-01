import torch
from ray import RayBatch


class Surface:
    def __init__(self, x_func, y_func, t_min, t_max, n_front=1.0, n_back=1.5):
        self.x_func = x_func
        self.y_func = y_func
        self.t_min = t_min
        self.t_max = t_max
        self.n_front = float(n_front)
        self.n_back = float(n_back)

    def points(self, t: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.x_func(t), self.y_func(t)), dim=-1)

    def tangent(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clone().detach().requires_grad_(True)
        x = self.x_func(t)
        y = self.y_func(t)
        dx = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
        dy = torch.autograd.grad(y.sum(), t, create_graph=True)[0]
        tan = torch.stack((dx, dy), dim=-1)
        return tan / (torch.norm(tan, dim=-1, keepdim=True) + 1e-12)

    def normal_at(self, hitpoints: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def intersect(self, rays):
        raise NotImplementedError


class LineSurface(Surface):
    def __init__(self, x_pos, y_min=-5, y_max=5, n_front=1.0, n_back=1.5):
        def x_func(t):
            return torch.full_like(t, float(x_pos))

        def y_func(t):
            return t

        super().__init__(x_func, y_func, y_min, y_max, n_front, n_back)
        self.x_pos = torch.tensor(x_pos, dtype=torch.float64)

    def normal_at(self, hitpoints: torch.Tensor) -> torch.Tensor:
        N = hitpoints.shape[0]
        normal = torch.zeros((N, 2), dtype=torch.float64)
        normal[:, 0] = -1.0
        return normal

    def intersect(self, rays):
        pos_x, pos_y = rays.pos[:, 0], rays.pos[:, 1]
        dir_x, dir_y = rays.dir[:, 0], rays.dir[:, 1]

        t = (self.x_pos - pos_x) / (dir_x + 1e-12)
        valid = t > 0
        hit_x = pos_x + t * dir_x
        hit_y = pos_y + t * dir_y
        hits = torch.stack((hit_x, hit_y), dim=-1)
        hits[~valid] = torch.nan
        return hits


class ParametricSurface(Surface):
    def __init__(self, x_func, y_func, t_min, t_max, n_front=1.0, n_back=1.5):
        super().__init__(x_func, y_func, t_min, t_max, n_front, n_back)

    def normal_at(self, hitpoints: torch.Tensor) -> torch.Tensor:
        """
        Compute normals via local tangent of the parametric curve.
        We approximate t from y position since our y_func(t)≈t here,
        then rotate tangent 90° CCW.
        """
        t_approx = hitpoints[:, 1].detach().clone().requires_grad_(True)
        tan = self.tangent(t_approx)
        # Rotate tangent 90°: (dx,dy)->(-dy,dx)
        normals = torch.stack((-tan[:, 1], tan[:, 0]), dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)
        return normals

    def intersect(self, rays):
        # iterative refinement: coarse x‑scan + local Newton step
        N, M = rays.pos.shape[0], 200
        t_samples = torch.linspace(self.t_min, self.t_max, M)
        curve = self.points(t_samples)  # (M, 2)

        pos_x, pos_y = rays.pos[:, 0], rays.pos[:, 1]
        dir_x, dir_y = rays.dir[:, 0], rays.dir[:, 1]

        # project each curve sample into ray space
        # analytic t_r = (x_curve - x_pos) / dir_x
        t_r = (curve[:, 0][None, :] - pos_x[:, None]) / (dir_x[:, None] + 1e-12)
        y_r = pos_y[:, None] + t_r * dir_y[:, None]
        diff = (y_r - curve[:, 1][None, :]).abs()  # |y_ray - y_curve|
        idx = diff.argmin(dim=1)  # best curve sample

        hits = curve[idx]
        # mask hits behind the ray start
        miss = (curve[idx][:, 0] - pos_x) * dir_x < 0
        hits[miss] = torch.tensor([float("nan"), float("nan")], dtype=hits.dtype)
        return hits


def refract_batch(rays, normals, n1, n2):
    n_ratio = n1 / n2
    dirs = rays.dir
    dot_dn = torch.sum(dirs * normals, dim=-1, keepdim=True)
    normals = torch.where(dot_dn > 0, -normals, normals)
    cos_i = -(dirs * normals).sum(-1, keepdim=True)
    sin2_t = n_ratio**2 * (1.0 - cos_i**2)
    valid = sin2_t <= 1.0
    cos_t = torch.sqrt(torch.clamp(1.0 - sin2_t, min=0.0))
    refracted = n_ratio * dirs + (n_ratio * cos_i - cos_t) * normals
    refracted = refracted / (torch.norm(refracted, dim=-1, keepdim=True) + 1e-12)
    return RayBatch(rays.pos, refracted), valid.squeeze(-1)


class Lens:
    def __init__(self, front, back):
        self.front_surface = front
        self.back_surface = back
