import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class LensVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.colors = {
            "incident": "red",
            "in_glass": "orange",
            "exit": "green",
            "reflected": "purple",
        }

    def draw(self, ray_paths, surfaces, show=True):
        # draw parametric surfaces
        for i, s in enumerate(surfaces):
            ts = np.linspace(s.t_min, s.t_max, 400)
            x = [s.x_func(t) for t in ts]
            y = [s.y_func(t) for t in ts]
            label = "Front Surface" if i == 0 else "Back Surface"
            self.ax.plot(x, y, color="black", lw=2, label=label)

        # draw rays
        for path in ray_paths:
            for seg in path:
                state = seg["state"]
                color = self.colors.get(state, "gray")
                self.ax.plot(
                    [seg["start"].x, seg["end"].x],
                    [seg["start"].y, seg["end"].y],
                    color=color,
                    lw=1.4,
                )

        # formatting
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_title("Lens Ray Tracing Simulation")
        self.ax.axis("equal")
        self.ax.grid(alpha=0.3)

        handles = [
            Line2D([0], [0], color="black", lw=2, label="Front Surface"),
            Line2D([0], [0], color="black", lw=2, label="Back Surface"),
        ] + [
            Line2D([0], [0], color=c, lw=2, label=n)
            for n, c in {
                "Incident Rays": "red",
                "In‑Glass Rays": "orange",
                "Exit Rays": "green",
                "Reflected Rays": "purple",
            }.items()
        ]
        self.ax.legend(handles=handles, loc="upper right")
        plt.tight_layout()
        if show:
            plt.show()
