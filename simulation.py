import torch
from surface import refract_batch


class Simulation:
    def __init__(self, rays, lenses):
        # Initial conditions and optical setup
        self.rays = rays  # Incoming batch of parallel or arbitrary rays
        self.lenses = lenses  # List of Lens objects (each has front/back surfaces)
        self.paths = []  # Used by the visualizer to draw all ray segments

    def run(self):
        rays = self.rays
        self.paths = []

        # Process rays through each lens sequentially
        for lens in self.lenses:
            # Front surface intersection
            hits_front = lens.front_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_front).any(dim=-1)  # valid intersections only

            # Segment before entering the lens (air -> lens interface)
            pre_seg_end = hits_front[mask_ok]
            for s, e in zip(rays.pos[mask_ok], pre_seg_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

            # Keep only rays that actually hit the front surface
            rays.pos = pre_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break  # all rays missed or were invalid

            # Refraction at front surface (into lens material)
            normals_front = lens.front_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_front,
                lens.front_surface.n_front,  # n before surface
                lens.front_surface.n_back,  # n inside lens
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]

            # Back surface intersection (ray inside lens hitting back face)
            hits_back = lens.back_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_back).any(dim=-1)

            # Segment inside the lens (glass path)
            inside_seg_end = hits_back[mask_ok]
            for s, e in zip(rays.pos[mask_ok], inside_seg_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

            # Keep only rays that reach the back surface
            rays.pos = inside_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break

            # Refraction at back surface (exit into air)
            normals_back = lens.back_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_back,
                lens.back_surface.n_front,  # n inside lens
                lens.back_surface.n_back,  # n after lens (air)
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]

            # Extend each exiting ray forward for visualization
            exit_end = rays.pos + rays.dir * 40.0
            for s, e in zip(rays.pos, exit_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

        return rays
