import torch
from ray import RayBatch
from surface import LineSurface, ParametricSurface, Lens
from simulation import Simulation
from visualizer import Visualizer


def straight():
    N = 4000
    y = torch.linspace(-5, 5, N)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]]).repeat(N, 1),
    )

    front = LineSurface(5.0, -5, 5, 1.0, 1.5)
    back = LineSurface(9.0, -5, 5, 1.5, 1.0)
    sim = Simulation(rays, [Lens(front, back)])
    out = sim.run()

    print(f"{out.pos.shape[0]} rays exited the system.")

    viz = Visualizer()
    viz.plot_surface(front, color="black", label="Front Surface")
    viz.plot_surface(back, color="black", label="Back Surface")
    viz.plot_rays(sim.paths, alpha=0.5)
    viz.show()


def curved():
    N = 50000
    y = torch.linspace(-5, 5, N)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]]).repeat(N, 1),
    )

    # --- Front Surface (flat) ---
    front = LineSurface(5.0, -5, 5, 1.0, 1.5)

    # --- Back Surface (curved parametric) ---
    # Example curve: bulges right → convex back surface
    def x_func(t):
        # Gaussian bump centered at y=0, convex toward +x
        return 9.0 + 1.0 * torch.exp(-((t / 3.0) ** 2))

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -5, 5, 1.5, 1.0)

    sim = Simulation(rays, [Lens(front, back)])
    out = sim.run()

    print(f"{out.pos.shape[0]} rays exited the system.")

    viz = Visualizer()
    viz.plot_surface(front, color="black", label="Front Surface")
    viz.plot_surface(back, color="black", label="Curved Back Surface")
    viz.plot_rays(sim.paths, alpha=0.5)
    viz.show()


def focus():
    N = 50
    y = torch.linspace(-4, 4, N)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]]).repeat(N, 1),
    )

    # --- Lens parameters ---
    n_lens = 1.5
    x_front = 5.0  # position of flat front surface
    x0 = 9.0  # reference point of back surface
    f = 25.0  # desired focus position (where rays converge)

    # --- Front Surface (flat) ---
    front = LineSurface(x_front, -4, 4, 1.0, n_lens)

    # --- Back Surface (parametric, focusing) ---
    def x_func(t):
        under = (f - x_front) ** 2 + (n_lens**2 - 1) * (t**2)
        under = torch.clamp(under, min=0.0)
        return x_front + (n_lens * (f - x_front) - torch.sqrt(under)) / (n_lens**2 - 1)

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -4, 4, n_lens, 1.0)

    # --- Simulation ---
    sim = Simulation(rays, [Lens(front, back)])
    out = sim.run()

    print(
        f"{out.pos.shape[0]} rays exited the system (should all intersect near x={f})."
    )

    # --- Visualization ---
    viz = Visualizer()
    viz.plot_surface(front, color="black", label="Front Surface")
    viz.plot_surface(back, color="black", label="Focusing Back Surface")
    viz.plot_rays(sim.paths, alpha=0.5)
    viz.show()


if __name__ == "__main__":
    focus()
