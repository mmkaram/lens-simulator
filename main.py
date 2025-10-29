import numpy as np
from vec2 import Vec2
from surface import LineSurface, ParametricSurface, MeasurementSurface
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


def test_detector():
    n_lens = 1.5
    x_front_pos = 5.0
    x_focus = 25.0
    aperture_y = 4.0

    front = LineSurface(x_front_pos, -aperture_y, aperture_y, 1.0, n_lens)

    def x_back(y):
        x0 = 9.0
        f, n = x_focus, n_lens
        under = max((f - x0) ** 2 - (n**2 - 1) * y**2, 0)
        return (n**2 * (f - x0) + n * math.sqrt(under)) / (n**2 - 1) + x0

    def y_func(t):
        return t

    back = ParametricSurface(x_back, y_func, -aperture_y, aperture_y, n_lens, 1.0)
    lens = Lens(front, back)
    rays = [
        Ray(Vec2(0, y), Vec2(1, 0)) for y in np.linspace(-aperture_y, aperture_y, 400)
    ]

    detector = MeasurementSurface(x_pos=26.0, y_min=-5, y_max=5, bins=150)

    print("Starting sim")
    sim = Simulation(rays, [lens])
    ray_paths, _ = sim.run()

    print("Starting viz")
    viz = LensVisualizer()
    viz.draw(ray_paths, [front, back])

    print("Starting density sim")
    hits = [rp[-1]["end"] for rp in ray_paths if rp]
    detector.record(hits)
    detector.plot_profile()


if __name__ == "__main__":
    # convex_back_lens()
    # concave_back_lens()
    test_detector()
