from abc import ABC, abstractmethod
import numpy as np
from vec2 import Vec2


class Side:
    FRONT = "front"
    BACK = "back"


class OpticalSurface(ABC):
    """Abstract base class for any optical surface"""

    def __init__(self, n_front, n_back):
        self.n_front = n_front
        self.n_back = n_back

    @abstractmethod
    def intersect(self, ray) -> Vec2 | None:
        """Return intersection point Vec2, or None if no intersection"""
        pass

    @abstractmethod
    def normal_at(self, point: Vec2) -> Vec2:
        """Return unit surface normal at the given point"""
        pass

    def refractive_index(self, side: Side) -> float:
        return self.n_front if side == Side.FRONT else self.n_back


class FlatSurface(OpticalSurface):
    def __init__(self, point: Vec2, normal: Vec2, n_front: float, n_back: float):
        super().__init__(n_front, n_back)
        self.point = point
        self.normal = normal.normalize()

    def intersect(self, ray):
        denom = ray.direction.dot(self.normal)
        if abs(denom) < 1e-10:
            return None

        t = (self.point.sub(ray.position)).dot(self.normal) / denom
        if t > 1e-10:
            return ray.position.add(ray.direction.scale(t))
        return None

    def normal_at(self, point):
        return self.normal


import torch
from vec2 import Vec2
from surface import OpticalSurface, Side


class CircularSurface(OpticalSurface):
    def __init__(self, center: Vec2, radius: float, n_front: float, n_back: float):
        super().__init__(n_front, n_back)
        self.center = center
        self.radius = radius

    def intersect(self, ray):
        # Ray-circle intersection
        oc = ray.position.sub(self.center)
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        disc = b * b - 4 * a * c

        if disc < 0:
            return None

        sqrt_disc = torch.sqrt(torch.tensor(disc))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t = None
        if t1 > 1e-10:
            t = t1
        elif t2 > 1e-10:
            t = t2
        else:
            return None

        return ray.position.add(ray.direction.scale(t.item()))

    def normal_at(self, point):
        # Normal is from center to point
        normal = point.sub(self.center).normalize()
        return normal  # direction will be fixed dynamically


class ParametricSurface(OpticalSurface):
    def __init__(self, x_func, y_func, t_min, t_max, n_front, n_back):
        super().__init__(n_front, n_back)
        self.x_func = x_func
        self.y_func = y_func
        self.t_min = t_min
        self.t_max = t_max

    def point_at(self, t):
        return Vec2(self.x_func(t), self.y_func(t))

    def intersect(self, ray):
        # Sample t and find closest intersection via distance minimization
        ts = np.linspace(self.t_min, self.t_max, 500)
        min_t, best_p = None, None
        min_dist = 1e9
        for t in ts:
            p = self.point_at(t)
            to_p = p.sub(ray.position)
            proj = to_p.dot(ray.direction)
            if proj <= 0:
                continue
            closest = ray.position.add(ray.direction.scale(proj))
            dist = (p.sub(closest)).length()
            if dist < min_dist:
                min_dist, min_t, best_p = dist, t, p
        if min_dist < 1e-1:  # or even 5e-2
            return best_p
        return None

    def normal_at(self, point):
        dt = 1e-4
        ts = np.linspace(self.t_min, self.t_max, 200)
        t_best = min(ts, key=lambda t: self.point_at(t).sub(point).length())
        dx = self.x_func(t_best + dt) - self.x_func(t_best - dt)
        dy = self.y_func(t_best + dt) - self.y_func(t_best - dt)
        tangent = Vec2(dx, dy).normalize()
        return Vec2(-tangent.y, tangent.x)
