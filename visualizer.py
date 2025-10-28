import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class LensVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # Ray colors for path states
        self.colors = {
            "incident": "red",
            "in_glass": "orange",
            "exit": "green",
            "reflected": "purple",
        }

    def draw(self, ray_paths, surfaces, show=True):
        # --- Draw surfaces ---
        for idx, surf in enumerate(surfaces):
            if surf["type"] == "flat":
                self.ax.axvline(
                    x=surf["x"],
                    color="black",
                    linewidth=2,
                    label="Front Surface" if idx == 0 else None,
                )
            elif surf["type"] == "circular":
                self.ax.plot(
                    surf["x"],
                    surf["y"],
                    color="black",
                    linewidth=2,
                    label="Back Surface" if idx == 1 else None,
                )

        # --- Draw rays ---
        for path in ray_paths:
            for seg in path:
                color = self.colors.get(seg["state"], "gray")
                self.ax.plot(
                    [seg["start"].x, seg["end"].x],
                    [seg["start"].y, seg["end"].y],
                    color=color,
                    linewidth=1.3,
                )

        # --- Axis and formatting ---
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_title("Lens Ray Tracing Simulation")
        self.ax.axis("equal")
        self.ax.grid(True, alpha=0.3)

        # --- Custom Legend ---
        # Create legend handles for the ray colors and surfaces
        custom_lines = [
            Line2D([0], [0], color="black", lw=2, label="Front Surface"),
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Back Surface"),
            Line2D(
                [0], [0], color=self.colors["incident"], lw=2, label="Incident Rays"
            ),
            Line2D(
                [0], [0], color=self.colors["in_glass"], lw=2, label="In Glass Rays"
            ),
            Line2D([0], [0], color=self.colors["exit"], lw=2, label="Exit Rays"),
            Line2D(
                [0], [0], color=self.colors["reflected"], lw=2, label="Reflected Rays"
            ),
        ]

        self.ax.legend(handles=custom_lines, loc="upper right")
        plt.tight_layout()

        # --- Show ---
        if show:
            plt.show()
