import torch
from ray import RayBatch
from surface import LineSurface, ParametricSurface, Lens
from simulation import Simulation
from visualizer import Visualizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PREFER = "gpu"

if PREFER.lower() == "gpu" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running on device: {device}")


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
    N = 1000000
    y = torch.linspace(-4, 4, N, device=device)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]], device=device).repeat(N, 1),
    ).to(device)

    n_lens = 1.5
    x_front = 5.0
    f = 25.0

    front = LineSurface(x_front, -4, 4, 1.0, n_lens)
    back = ParametricSurface(
        lambda t: x_front
        + (
            n_lens * (f - x_front)
            - torch.sqrt(
                torch.clamp((f - x_front) ** 2 + (n_lens**2 - 1) * (t**2), min=0.0)
            )
        )
        / (n_lens**2 - 1),
        lambda t: t,
        -4,
        4,
        n_lens,
        1.0,
    )

    lens = Lens(front, back).to(device)
    sim = Simulation(rays, [lens]).to(device)
    out = sim.run()

    print(f"{out.pos.shape[0]} rays exited (expected near focus x={f}).")

    # viz = Visualizer()
    # viz.plot_surface(front, color="black", label="Front Surface")
    # viz.plot_surface(back, color="black", label="Focusing Back Surface")
    # viz.plot_rays(sim.paths, alpha=0.5)
    # viz.show()


if __name__ == "__main__":
    focus()
