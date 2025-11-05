import torch
from surface import refract_batch


class Simulation:
    def __init__(self, rays, lenses):
        # Initial conditions and optical setup
        self.rays = rays  # Incoming batch of parallel or arbitrary rays
        self.lenses = lenses  # List of Lens objects (each has front/back surfaces)
        self.paths = []  # Used by the visualizer to draw all ray segments
        self.device = self.rays.pos.device  # automatically track compute device

    def to(self, device):
        """Move simulation, rays, and lenses to a given device."""
        self.device = torch.device(device)
        if hasattr(self.rays, "to"):
            self.rays.to(self.device)
        for lens in self.lenses:
            if hasattr(lens, "to"):
                lens.to(self.device)
        return self

    def run(self):
        rays = self.rays
        self.paths = []

        # Ensure all lenses are on the same device as rays
        for lens in self.lenses:
            lens.to(rays.pos.device)

        # Process rays through each lens sequentially
        for lens in self.lenses:
            # --- Front surface intersection ---
            hits_front = lens.front_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_front).any(dim=-1)

            # Record segment (air → front surface)
            pre_seg_end = hits_front[mask_ok]
            # Store CPU copies only for visualization
            for s, e in zip(rays.pos[mask_ok], pre_seg_end):
                self.paths.append((s.detach().to("cpu"), e.detach().to("cpu")))

            # Keep only rays that hit the front surface
            rays.pos = pre_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break  # No valid intersections remain

            # --- Refraction: air → lens ---
            normals_front = lens.front_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_front,
                lens.front_surface.n_front,
                lens.front_surface.n_back,
            )
            # filter out invalid rays
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]
            if len(rays.pos) == 0:
                break

            # --- Inside lens intersection (front ➜ back) ---
            hits_back = lens.back_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_back).any(dim=-1)

            inside_seg_end = hits_back[mask_ok]
            for s, e in zip(rays.pos[mask_ok], inside_seg_end):
                self.paths.append((s.detach().to("cpu"), e.detach().to("cpu")))

            rays.pos = inside_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break

            # --- Refraction: lens → air ---
            normals_back = lens.back_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_back,
                lens.back_surface.n_front,
                lens.back_surface.n_back,
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]
            if len(rays.pos) == 0:
                break

            # --- Extend rays after exiting lens (for viz only) ---
            exit_end = rays.pos + rays.dir * 40.0
            for s, e in zip(rays.pos, exit_end):
                # Only convert to CPU for drawing to avoid GPU→CPU overhead mid-sim
                self.paths.append((s.detach().to("cpu"), e.detach().to("cpu")))

        return rays
