import torch
from surface import refract_batch


class Simulation:
    def __init__(self, rays, lenses, measurement=None):
        """
        rays: RayBatch
        lenses: list[ Lens ]
        measurement: optional MeasurementSurface to record ray impact density
        """
        self.rays = rays
        self.lenses = lenses
        self.measurement = measurement
        self.paths = []
        self.device = self.rays.pos.device

    def to(self, device):
        self.device = torch.device(device)
        if hasattr(self.rays, "to"):
            self.rays.to(self.device)
        for lens in self.lenses:
            if hasattr(lens, "to"):
                lens.to(self.device)
        if hasattr(self.measurement, "to"):
            self.measurement.to(self.device)
        return self

    def run(self):
        rays = self.rays
        self.paths = []

        # Run through all lenses
        for lens in self.lenses:
            lens.to(rays.pos.device)

            # FRONT FACE
            hits_front = lens.front_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_front).any(dim=-1)
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

            # BACK FACE
            hits_back = lens.back_surface.intersect(rays)
            mask_ok = ~torch.isnan(hits_back).any(dim=-1)
            inside_seg_end = hits_back[mask_ok]
            for s, e in zip(rays.pos[mask_ok], inside_seg_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

            rays.pos = inside_seg_end
            rays.dir = rays.dir[mask_ok]
            if len(rays.pos) == 0:
                break

            normals_back = lens.back_surface.normal_at(rays.pos)
            rays, valid = refract_batch(
                rays,
                normals_back,
                lens.back_surface.n_front,
                lens.back_surface.n_back,
            )
            rays.pos = rays.pos[valid]
            rays.dir = rays.dir[valid]

            exit_end = rays.pos + rays.dir * 40.0
            for s, e in zip(rays.pos, exit_end):
                self.paths.append((s.detach().cpu(), e.detach().cpu()))

        # Optional: measure ray impact distribution
        if self.measurement is not None:
            self.measurement.measure_density(rays)

        return rays
