import numpy as np
from vec2 import Vec2
from surface import LineSurface, SemiCircleSurface, ParametricSurface
from lens import Lens
from ray import Ray
from simulation import Simulation
from visualizer import LensVisualizer
import math


def main():
    front = LineSurface(x_pos=5.0, y_min=-5.0, y_max=5.0, n_front=1.0, n_back=1.5)

    # back = SemiCircleSurface(Vec2(10, 0), 6, direction="right", n_front=1.5, n_back=1.0)

    def x_func(t):
        return 9.0 + 1.5 * math.exp(-((t / 3.0) ** 2))

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -5, 5, n_front=1.5, n_back=1.0)

    lens = Lens(front, back)
    rays = [Ray(Vec2(0, y), Vec2(1, 0)) for y in np.linspace(-5, 5, 20)]

    sim = Simulation(rays, [lens])
    paths, _ = sim.run()
    viz = LensVisualizer()
    viz.draw(paths, [front, back])


if __name__ == "__main__":
    main()
