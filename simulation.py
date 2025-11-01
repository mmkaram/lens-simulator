import torch
from surface import refract_batch


class Simulation:
    def __init__(self, rays, lenses):
        self.rays = rays
        self.lenses = lenses
        self.paths = []  # stores (x_start, y_start) -> (x_end, y_end)

    def run(self):
        rays = self.rays
        self.paths = []
        for lens in self.lenses:
            # --- Front surface ---
            hits_front = lens.front_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_front).any(dim=-1)
            # store pre-lens segment
            pre_seg_end = hits_front[mask_ok]
            for s, e in zip(rays.pos[mask_ok], pre_seg_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))
            rays.pos = pre_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break

            normals_front = lens.front_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_front,
                lens.front_surface.n_front,
                lens.front_surface.n_back,
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]

            # --- Back surface ---
            hits_back = lens.back_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_back).any(dim=-1)
            # store inside-lens segment
            inside_seg_end = hits_back[mask_ok]
            for s, e in zip(rays.pos[mask_ok], inside_seg_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))
            rays.pos = inside_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break

            normals_back = lens.back_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays, normals_back, lens.back_surface.n_front, lens.back_surface.n_back
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]

            # store exit segment
            exit_end = rays.pos + rays.dir * 10.0
            for s, e in zip(rays.pos, exit_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

        return rays
