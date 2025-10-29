import math


class Ray:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction.normalize()

    def refract(self, intersection, normal, n1, n2):
        """
        Apply Snell's law to compute refraction direction.
        Args:
            intersection: Vec2 — the point of intersection
            normal: Vec2 — surface normal (any orientation)
            n1: refractive index of incoming medium
            n2: refractive index of outgoing medium
        Returns:
            New Ray or None if total internal reflection
        """

        # Ensure the normal points *against* the incident ray
        if normal.dot(self.direction) > 0:
            normal = normal.scale(-1.0)

        # Snell's law
        n_ratio = n1 / n2
        cos_i = -normal.dot(self.direction)
        sin2_t = n_ratio**2 * (1.0 - cos_i**2)

        # Handle total internal reflection
        if sin2_t > 1.0:
            return None

        cos_t = math.sqrt(1.0 - sin2_t)

        # Compute refracted direction (vector form of Snell's law)
        refracted = (
            self.direction.scale(n_ratio)
            .add(normal.scale(n_ratio * cos_i - cos_t))
            .normalize()
        )

        return Ray(intersection, refracted)
