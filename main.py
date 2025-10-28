from vec2 import Vec2
from ray import Ray
from surface import FlatSurface, CircularSurface
from surface import ParametricSurface
from lens import Lens
from simulation import Simulation
from visualizer import LensVisualizer
import math


def x_func(t):
    # Gaussian bump that bulges to the right, away from the front surface
    return 8.0 + 1.5 * math.exp(-((t / 3.0) ** 2))


def y_func(t):
    return t


def main():
    # Define lens
    front = FlatSurface(Vec2(5, 0), Vec2(-1, 0), n_front=1.0, n_back=1.5)
    back = CircularSurface(Vec2(15, 0), 9.0, n_front=1.5, n_back=1.0)
    back_surface = ParametricSurface(
        x_func=x_func,
        y_func=y_func,
        t_min=-5.0,
        t_max=5.0,
        n_front=1.5,
        n_back=1.0,
    )
    lens = Lens(front, back_surface)

    # Rays
    rays = [
        Ray(Vec2(0.0, y), Vec2(1.0, 0.0)) for y in [i * 0.5 - 5.0 for i in range(20)]
    ]

    # Run pure math
    sim = Simulation(rays, [lens])
    ray_paths, surfaces = sim.run()
    print(sim.get_statistics())

    # Visualize separately
    viz = LensVisualizer()
    viz.draw(ray_paths, surfaces)


if __name__ == "__main__":
    main()
