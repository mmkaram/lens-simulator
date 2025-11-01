import torch
import matplotlib.pyplot as plt
from surface import Surface


class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def plot_surface(self, surface: Surface, color="black", label=None):
        t_vals = torch.linspace(surface.t_min, surface.t_max, 200)
        pts = surface.points(t_vals)
        self.ax.plot(
            pts[:, 0].detach(),
            pts[:, 1].detach(),
            color=color,
            lw=2,
            label=label,
        )

    def plot_rays(self, paths, color="orange", alpha=0.5):
        for s, e in paths:
            self.ax.plot(
                [s[0], e[0]],
                [s[1], e[1]],
                color=color,
                lw=0.8,
                alpha=alpha,
            )

    def show(self):
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.set_title("Ray trace through lens system")
        self.ax.axis("equal")
        self.ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
