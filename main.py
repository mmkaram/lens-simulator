import numpy as np
from vec2 import Vec2
from surface import LineSurface, ParametricSurface
from lens import Lens
from ray import Ray
from simulation import Simulation
from visualizer import LensVisualizer
import math


def sim_helper(front, back, rays):
    """
    Runs a lens simulation given front/back surfaces and rays.
    Handles the creation of the Lens, Simulation, and Visualization.
    """
    lens = Lens(front, back)
    sim = Simulation(rays, [lens])
    paths, _ = sim.run()

    viz = LensVisualizer()
    viz.draw(paths, [front, back])


def convex_back_lens():
    """Lens with convex (bulging right) back surface."""
    front = LineSurface(x_pos=5.0, y_min=-5.0, y_max=5.0, n_front=1.0, n_back=1.5)

    def x_func(t):
        # bulges right → convex (bulging away from incoming light)
        return 9.0 + 1.5 * math.exp(-((t / 3.0) ** 2))

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -5, 5, n_front=1.5, n_back=1.0)
    rays = [Ray(Vec2(0, y), Vec2(1, 0)) for y in np.linspace(-5, 5, 20)]

    sim_helper(front, back, rays)


def concave_back_lens():
    """Lens with concave (curving toward incoming light) back surface."""
    front = LineSurface(x_pos=5.0, y_min=-5.0, y_max=5.0, n_front=1.0, n_back=1.5)

    def x_func(t):
        # bulges left → concave (facing incoming light)
        return 9.0 - 1.5 * math.exp(-((t / 3.0) ** 2))

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -5, 5, n_front=1.5, n_back=1.0)
    rays = [Ray(Vec2(0, y), Vec2(1, 0)) for y in np.linspace(-5, 5, 20)]

    sim_helper(front, back, rays)


if __name__ == "__main__":
    convex_back_lens()
    # concave_back_lens()   # uncomment to test this one
