from vec2 import Vec2
from ray import Ray
from surface import Side


class Simulation:
    def __init__(self, rays, lenses):
        self.rays = rays
        self.lenses = lenses
        self.ray_paths = []
        self.surfaces = []

    def trace_ray_through_lens(self, ray, lens, ray_index=None):
        segments = []
        print(f"\n[TRACE] Ray {ray_index if ray_index is not None else '?'}")
        print(
            f"  Start position=({ray.position.x:.3f}, {ray.position.y:.3f}) "
            f"dir=({ray.direction.x:.3f}, {ray.direction.y:.3f})"
        )

        # --- Front Surface ---
        intersection1 = lens.front_surface.intersect(ray)
        if not intersection1:
            print("  X - No intersection with front surface.")
            return segments

        print(
            f"  → Hit front surface at ({intersection1.x:.3f}, {intersection1.y:.3f})"
        )
        segments.append(
            {"start": ray.position, "end": intersection1, "state": "incident"}
        )

        normal1 = lens.front_surface.normal_at(intersection1)
        n1 = lens.front_surface.refractive_index(Side.FRONT)
        n2 = lens.front_surface.refractive_index(Side.BACK)
        print(
            f"  n1={n1:.3f}, n2={n2:.3f}, front normal=({normal1.x:.3f}, {normal1.y:.3f})"
        )

        ray_in_glass = ray.refract(intersection1, normal1, n1, n2)
        if not ray_in_glass:
            print("  ! - Total internal reflection at front surface.")
            segments.append(
                {"start": intersection1, "end": intersection1, "state": "reflected"}
            )
            return segments
        print(
            f"  ↳ Refracted into glass, dir=({ray_in_glass.direction.x:.3f}, {ray_in_glass.direction.y:.3f})"
        )

        # --- Back Surface ---
        intersection2 = lens.back_surface.intersect(ray_in_glass)
        if not intersection2:
            print("  ! - Missed back surface — continuing straight out.")
            miss = ray_in_glass.position.add(ray_in_glass.direction.scale(20.0))
            segments.append(
                {"start": ray_in_glass.position, "end": miss, "state": "exit"}
            )
            return segments

        print(f"  → Hit back surface at ({intersection2.x:.3f}, {intersection2.y:.3f})")
        segments.append(
            {"start": intersection1, "end": intersection2, "state": "in_glass"}
        )

        normal2 = lens.back_surface.normal_at(intersection2)
        n3 = lens.back_surface.refractive_index(Side.FRONT)
        n4 = lens.back_surface.refractive_index(Side.BACK)
        print(
            f"  n3={n3:.3f}, n4={n4:.3f}, back normal=({normal2.x:.3f}, {normal2.y:.3f})"
        )

        exit_ray = ray_in_glass.refract(intersection2, normal2, n3, n4)
        if exit_ray:
            print(
                f"  ↳ Exit dir=({exit_ray.direction.x:.3f}, {exit_ray.direction.y:.3f})"
            )
            exit_end = exit_ray.position.add(exit_ray.direction.scale(20.0))
            segments.append({"start": intersection2, "end": exit_end, "state": "exit"})
        else:
            print("  ! - Total internal reflection at back surface.")
            segments.append(
                {"start": intersection2, "end": intersection2, "state": "reflected"}
            )

        return segments

    # Full sim
    def run(self):
        self.ray_paths.clear()
        for i, ray in enumerate(self.rays):
            segments_total = []
            current = ray
            for lens in self.lenses:
                segs = self.trace_ray_through_lens(current, lens, ray_index=i)
                segments_total.extend(segs)
                if segs and segs[-1]["state"] == "exit":
                    last = segs[-1]
                    direction = last["end"].sub(last["start"]).normalize()
                    current = Ray(last["end"], direction)
                else:
                    break
            self.ray_paths.append(segments_total)

        self._build_surface_data()
        return self.ray_paths, self.surfaces

    # Surface coord selection
    def _build_surface_data(self):
        import numpy as np

        self.surfaces = []

        for lens in self.lenses:
            for surf, label in [
                (lens.front_surface, "front"),
                (lens.back_surface, "back"),
            ]:
                ts = np.linspace(surf.t_min, surf.t_max, 400)
                x_vals = [surf.x_func(t) for t in ts]
                y_vals = [surf.y_func(t) for t in ts]
                self.surfaces.append(
                    {
                        "type": "parametric",
                        "which": label,
                        "x": np.array(x_vals),
                        "y": np.array(y_vals),
                        "surface": surf,
                    }
                )

    # Summary
    def get_statistics(self):
        total = len(self.rays)
        complete = sum(1 for p in self.ray_paths if p and p[-1]["state"] == "exit")
        reflected = sum(
            1 for p in self.ray_paths if any(s["state"] == "reflected" for s in p)
        )
        return {
            "total_rays": total,
            "completed_rays": complete,
            "reflected_rays": reflected,
            "completion_rate": complete / total if total else 0,
        }
