import torch
from ray import RayBatch
from surface import LineSurface, ParametricSurface, Lens, MeasurementSurface
from simulation import Simulation
from visualizer import Visualizer

PREFER = "cpu"

if PREFER.lower() == "gpu" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running on device: {device}")


def run_simulation(rays, lens, screen, label_front, label_back):
    """Helper to run a simulation and visualize results."""
    sim = Simulation(rays, [lens], measurement=screen).to(device)
    out = sim.run()

    print(f"{out.pos.shape[0]} rays exited the system.")
    print("Simulation complete – visualizing...")

    # Visualization setup
    viz = Visualizer()

    # Plot optical elements
    viz.plot_surface(lens.front_surface, color="black", label=label_front)
    viz.plot_surface(lens.back_surface, color="black", label=label_back)

    # Plot the measurement plane visually (as a dashed red line)
    y_line = torch.linspace(screen.t_min, screen.t_max, 2)
    screen_x = torch.full_like(y_line, screen.x_pos_value)
    viz.ax.plot(
        screen_x.cpu(),
        y_line.cpu(),
        color="red",
        linestyle="--",
        label="Measurement Plane",
    )

    # Plot rays
    viz.plot_rays(sim.paths, alpha=0.5)
    viz.ax.legend(loc="upper right")
    viz.show()

    # Plot density histogram for captured flux
    screen.measure_density(out)


def straight():
    N = 400
    y = torch.linspace(-5, 5, N, device=device)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]], device=device).repeat(N, 1),
    ).to(device)

    front = LineSurface(5.0, -5, 5, 1.0, 1.5)
    back = LineSurface(9.0, -5, 5, 1.5, 1.0)
    lens = Lens(front, back).to(device)

    screen = MeasurementSurface(
        x_pos=20.0, y_min=-5, y_max=5, bins=150, label="Straight Lens Density"
    )

    run_simulation(rays, lens, screen, "Front Surface", "Back Surface")


def curved():
    N = 500
    y = torch.linspace(-5, 5, N, device=device)
    rays = RayBatch(
        pos=torch.stack((torch.zeros_like(y), y), dim=-1),
        dir=torch.tensor([[1.0, 0.0]], device=device).repeat(N, 1),
    ).to(device)

    front = LineSurface(5.0, -5, 5, 1.0, 1.5)

    def x_func(t):
        return 9.0 + 1.0 * torch.exp(-((t / 3.0) ** 2))

    def y_func(t):
        return t

    back = ParametricSurface(x_func, y_func, -5, 5, 1.5, 1.0)
    lens = Lens(front, back).to(device)

    screen = MeasurementSurface(
        x_pos=25.0, y_min=-5, y_max=5, bins=150, label="Curved Lens Density"
    )

    run_simulation(rays, lens, screen, "Front Surface", "Curved Back Surface")


def focus():
    N = 100
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
    screen = MeasurementSurface(
        x_pos=50, y_min=-5, y_max=5, bins=150, label="Focusing Lens Density"
    )

    run_simulation(rays, lens, screen, "Front Surface", "Focusing Back Surface")


if __name__ == "__main__":
    straight()
    # curved()
    # focus()
