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

    def to(self, device):
        """Optional helper to move this surface to a device."""
        if hasattr(self, "x_pos") and isinstance(self.x_pos, torch.Tensor):
            self.x_pos = self.x_pos.to(device)
        return self


class LineSurface(Surface):
    def __init__(self, x_pos, y_min=-5, y_max=5, n_front=1.0, n_back=1.5):
        def x_func(t):
            return torch.full_like(t, float(x_pos), device=t.device)

        def y_func(t):
            return t

        super().__init__(x_func, y_func, y_min, y_max, n_front, n_back)
        # Do NOT fix device here — we'll move this dynamically on intersect
        self.x_pos_value = float(x_pos)
        self._device = None  # track last device

    def normal_at(self, hitpoints: torch.Tensor) -> torch.Tensor:
        N = hitpoints.shape[0]
        normal = torch.zeros((N, 2), dtype=torch.float64, device=hitpoints.device)
        normal[:, 0] = -1.0
        return normal

    def intersect(self, rays):
        # ensure we have a tensor version of x_pos on the same device and dtype as incoming rays
        device = rays.pos.device
        dtype = rays.pos.dtype
        self.x_pos = torch.tensor(
            self.x_pos_value,
            dtype=dtype,
            device=device,
        )

        pos_x, pos_y = rays.pos[:, 0], rays.pos[:, 1]
        dir_x, dir_y = rays.dir[:, 0], rays.dir[:, 1]

        t = (self.x_pos - pos_x) / (dir_x + 1e-12)
        valid = t > 0
        hit_x = pos_x + t * dir_x
        hit_y = pos_y + t * dir_y
        hits = torch.stack((hit_x, hit_y), dim=-1)
        hits[~valid] = torch.tensor(
            [float("nan"), float("nan")],
            dtype=hits.dtype,
            device=hits.device,
        )
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
        normals = torch.stack((-tan[:, 1], tan[:, 0]), dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)
        return normals

    def intersect(self, rays, max_iter=10, tol=1e-6):
        """
        Vectorized differentiable Newton iteration to find intersection of each ray
        with the parametric curve defined by x(t), y(t).
        """
        t = rays.pos[:, 1].clone().detach().requires_grad_(True)

        for _ in range(max_iter):
            x_t = self.x_func(t)
            y_t = self.y_func(t)

            f = y_t - (
                rays.pos[:, 1]
                + rays.dir[:, 1] * ((x_t - rays.pos[:, 0]) / (rays.dir[:, 0] + 1e-12))
            )

            df = torch.autograd.grad(f.sum(), t, create_graph=True)[0]

            t_next = t - f / (df + 1e-12)
            t = torch.clamp(t_next, self.t_min, self.t_max)

            if torch.all(torch.abs(f) < tol):
                break

        hit_x = self.x_func(t)
        hit_y = self.y_func(t)
        hits = torch.stack((hit_x, hit_y), dim=-1)

        t_ray = (hit_x - rays.pos[:, 0]) / (rays.dir[:, 0] + 1e-12)
        miss = t_ray < 0
        hits[miss] = torch.tensor(
            [float("nan"), float("nan")],
            dtype=hits.dtype,
            device=hits.device,
        )
        return hits


def refract_batch(rays, normals, n1, n2):
    # Ratio of refractive indices (n₁ / n₂)
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

    # make sure both pos and dir are on same device
    pos = rays.pos.to(normals.device)
    refracted = refracted.to(normals.device)

    return RayBatch(pos, refracted), valid.squeeze(-1)


class Lens:
    def __init__(self, front, back):
        self.front_surface = front
        self.back_surface = back

    def to(self, device):
        if hasattr(self.front_surface, "to"):
            self.front_surface.to(device)
        if hasattr(self.back_surface, "to"):
            self.back_surface.to(device)
        return self


import matplotlib.pyplot as plt


class MeasurementSurface(LineSurface):
    """
    A line surface that acts as a measurement plane — rays intersect it,
    and we record hit positions to estimate ray density (flux distribution).
    """

    def __init__(
        self,
        x_pos: float,
        y_min: float = -10.0,
        y_max: float = 10.0,
        bins: int = 200,
        label: str = "Measurement Plane",
        plot: bool = True,
    ):
        super().__init__(x_pos, y_min, y_max, n_front=1.0, n_back=1.0)
        self.bins = bins
        self.label = label
        self.plot_histogram = plot

    @torch.no_grad()
    def measure_density(self, rays):
        """Compute intersection y-positions and histogram them."""
        hits = self.intersect(rays)
        valid = ~torch.isnan(hits).any(dim=-1)
        hit_y = hits[valid, 1].detach().cpu()

        counts, edges = torch.histogram(
            hit_y,
            bins=self.bins,
            range=(self.t_min, self.t_max),
        )

        centers = 0.5 * (edges[1:] + edges[:-1])
        density = counts / counts.sum()  # normalize

        if self.plot_histogram:
            plt.figure(figsize=(6, 4))
            plt.plot(centers.numpy(), density.numpy(), color="blue")
            plt.title(self.label)
            plt.xlabel("y-position (on measurement plane)")
            plt.ylabel("Normalized ray density")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return centers, density
