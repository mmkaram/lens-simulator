from abc import ABC, abstractmethod
import numpy as np
import torch
from vec2 import Vec2


class Side:
    FRONT = "front"
    BACK = "back"


class OpticalSurface(ABC):
    """Any optical surface supporting intersection & normals."""

    def __init__(self, n_front: float, n_back: float):
        self.n_front = n_front
        self.n_back = n_back

    @abstractmethod
    def point_at(self, t: float) -> Vec2: ...
    @abstractmethod
    def intersect(self, ray) -> Vec2 | None: ...
    @abstractmethod
    def normal_at(self, point: Vec2) -> Vec2: ...

    def refractive_index(self, side: Side) -> float:
        return self.n_front if side == Side.FRONT else self.n_back


class ParametricSurface(OpticalSurface):
    def __init__(
        self, x_func, y_func, t_min: float, t_max: float, n_front: float, n_back: float
    ):
        super().__init__(n_front, n_back)
        self.x_func = x_func
        self.y_func = y_func
        self.t_min = t_min
        self.t_max = t_max

    # basic geometry sampling
    def point_at(self, t: float) -> Vec2:
        return Vec2(self.x_func(t), self.y_func(t))

    def tangent_at(self, t: float, dt: float = 1e-4) -> Vec2:
        dx = self.x_func(t + dt) - self.x_func(t - dt)
        dy = self.y_func(t + dt) - self.y_func(t - dt)
        return Vec2(dx, dy).normalize()

    # intersection solver
    def intersect(self, ray):
        ts = np.linspace(self.t_min, self.t_max, 700)
        best_p, best_d = None, float("inf")
        for t in ts:
            p = self.point_at(t)
            to_p = p.sub(ray.position)
            proj = to_p.dot(ray.direction)
            if proj <= 0:
                continue
            closest = ray.position.add(ray.direction.scale(proj))
            d = (p.sub(closest)).length()
            if d < best_d:
                best_d, best_p = d, p
        if best_d < 1e-1:
            return best_p
        return None

    # normal at surface
    def normal_at(self, point: Vec2) -> Vec2:
        ts = np.linspace(self.t_min, self.t_max, 200)
        t_best = min(ts, key=lambda t: self.point_at(t).sub(point).length())
        tangent = self.tangent_at(t_best)
        return Vec2(-tangent.y, tangent.x).normalize()


# Convinience shapes


class LineSurface(ParametricSurface):
    """Flat plane parameterized along y"""

    def __init__(
        self, x_pos: float, y_min: float, y_max: float, n_front: float, n_back: float
    ):
        def x_func(t):
            return x_pos

        def y_func(t):
            return t

        super().__init__(x_func, y_func, y_min, y_max, n_front, n_back)


class SemiCircleSurface(ParametricSurface):
    """Half circle described by center & radius."""

    def __init__(
        self,
        center: Vec2,
        radius: float,
        direction: str,  # "right" or "left"
        n_front: float,
        n_back: float,
    ):
        if direction not in ("right", "left"):
            raise ValueError("direction must be 'right' or 'left'")
        sign = 1 if direction == "right" else -1

        def x_func(theta):
            return center.x + sign * radius * np.cos(theta)

        def y_func(theta):
            return center.y + radius * np.sin(theta)

        if direction == "right":
            t_min, t_max = -np.pi / 2, np.pi / 2
        else:
            t_min, t_max = np.pi / 2, 3 * np.pi / 2

        super().__init__(x_func, y_func, t_min, t_max, n_front, n_back)
        self.center, self.radius, self.direction = center, radius, direction


class MeasurementSurface(ParametricSurface):
    """
    Virtual detector surface.
    Collects intersection points and computes
    ray density (irradiance proxy).
    """

    def __init__(self, x_pos: float, y_min: float, y_max: float, bins: int = 100):
        # simple flat parametric plane
        def x_func(t):
            return x_pos

        def y_func(t):
            return t

        super().__init__(x_func, y_func, y_min, y_max, n_front=1.0, n_back=1.0)
        self.bins = bins
        self.hits = np.zeros(bins)
        self.range = (y_min, y_max)

    def record(self, hit_points):
        ys = [p.y for p in hit_points]
        self.hits, edges = np.histogram(ys, bins=self.bins, range=self.range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.profile_y = centers
        self.profile_I = self.hits / np.max(self.hits)  # normalize

    def plot_profile(self):
        import matplotlib.pyplot as plt

        plt.plot(self.profile_y, self.profile_I)
        plt.title("Relative Irradiance Profile")
        plt.xlabel("Position (y)")
        plt.ylabel("Normalized Density (a.u.)")
        plt.show()
