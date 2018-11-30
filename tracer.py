from ray import Ray
from vector3 import Vector3


class Tracer:
    """Main (ray) tracer coordinating the heavy algorithmic work"""

    def __init__(self, max_recursion_depth=5, bias=1e-4):
        """Creates a new tracer"""
        self.__max_recursion_depth = max_recursion_depth
        self.__bias = bias

    def trace(self, ray, scene):
        """Traces a ray through a scene to return the traced color"""
        self.__scene = scene
        return self.__trace_recursively(ray, 0)

    def __trace_recursively(self, ray, depth):
        """Traces a ray through a scene recursively"""
        hit_object, hit_point, hit_normal = self.__intersect(ray)
        if hit_object is None:
            return Vector3(0.3, 0.3, 0.3)  # horizon
        traced_color = Vector3()
        if not hit_object.material.is_diffuse and depth < self.__max_recursion_depth:
            traced_color = self.__trace_non_diffuse(ray, hit_object, hit_point, hit_normal, depth)
        else:
            traced_color = self.__trace_diffuse(hit_object, hit_point, hit_normal)
        return traced_color + hit_object.material.emission_color

    def __intersect(self, ray):
        """Returns the (nearest) intersection of the ray"""
        hit_object = None
        hit_t, hit_point, hit_normal = float("inf"), None, None
        for obj in self.__scene:
            intersection = obj.primitive.intersect(ray)
            if intersection and intersection[0] < hit_t:
                hit_object = obj
                hit_t, hit_point, hit_normal = intersection
        return hit_object, hit_point, hit_normal

    def __trace_diffuse(self, hit_object, hit_point, hit_normal):
        """Traces color of an object with diffuse material"""
        summed_color = Vector3()
        for light in filter(lambda obj: obj.is_light, self.__scene):
            transmission = Vector3(1, 1, 1)
            light_direction = (light.primitive.position - hit_point).normalize()
            for other in filter(lambda obj: obj != light, self.__scene):
                if other.primitive.intersect(Ray(hit_point + self.__bias * hit_normal,
                                             light_direction)):
                    transmission = Vector3()
                    break
            summed_color = summed_color + (
                hit_object.material.surface_color
                .mul_comp(transmission)
                .mul_comp(light.material.emission_color) *
                max(0, hit_normal.dot(light_direction)))
        return summed_color

    def __trace_non_diffuse(self, ray, hit_object, hit_point, hit_normal, depth):
        """Traces color of an object with refractive/reflective material"""
        inside = ray.direction.dot(hit_normal) > 0
        if inside:
            hit_normal = -hit_normal
        facing_ratio = -ray.direction.dot(hit_normal)
        fresnel = self.__mix((1 - facing_ratio) ** 2, 1, 0.1)
        reflection_ray = Ray(hit_point + self.__bias * hit_normal,
                             ray.direction.reflect(hit_normal).normalize())
        reflection = self.__trace_recursively(reflection_ray, depth + 1)
        refraction = Vector3()

        # transparent?
        if hit_object.material.transparency > 0:
            from_ior = ray.current_ior if inside else hit_object.material.ior
            to_ior = hit_object.material.ior if inside else ray.current_ior
            refraction_ray = Ray(hit_point - self.__bias * hit_normal,
                                 ray.direction.refract(from_ior, to_ior, hit_normal)
                                 .normalize())
            refraction = self.__trace_recursively(refraction_ray, depth + 1)

        # mix according to fresnel
        return ((reflection * fresnel +
                refraction * (1 - fresnel) * hit_object.material.transparency)
                .mul_comp(hit_object.material.surface_color))

    def __mix(self, a, b, mix):
        """Mixes to values by a factor"""
        return b * mix + a * (1 - mix)

class Tracer_gpu:
    """Main (ray) tracer coordinating the heavy algorithmic work"""

    def __init__(self, max_recursion_depth=5, bias=1e-4):
        """Creates a new tracer"""
        self.__max_recursion_depth = max_recursion_depth
        self.__bias = bias

    def trace(self, ray_array, scene):
        """Traces a ray through a scene to return the traced color"""
        self.__scene = scene
        return self.__trace_recursively(ray, 0)

    def __trace_recursively(self, ray, depth):
        """Traces a ray through a scene recursively"""
        hit_object, hit_point, hit_normal = self.__intersect(ray)
        if hit_object is None:
            return Vector3(0.3, 0.3, 0.3)  # horizon
        traced_color = Vector3()
        if not hit_object.material.is_diffuse and depth < self.__max_recursion_depth:
            traced_color = self.__trace_non_diffuse(ray, hit_object, hit_point, hit_normal, depth)
        else:
            traced_color = self.__trace_diffuse(hit_object, hit_point, hit_normal)
        return traced_color + hit_object.material.emission_color

    def __intersect(self, ray):
        """Returns the (nearest) intersection of the ray"""
        hit_object = None
        hit_t, hit_point, hit_normal = float("inf"), None, None
        for obj in self.__scene:
            intersection = obj.primitive.intersect(ray)
            if intersection and intersection[0] < hit_t:
                hit_object = obj
                hit_t, hit_point, hit_normal = intersection
        return hit_object, hit_point, hit_normal

    def __intersect_gpu(self, ray):
        """Returns the (nearest) intersection of the ray"""
        hit_object = None
        hit_t, hit_point, hit_normal = float("inf"), None, None

        """
        scene -> obj array for gpu
        """

        for obj in self.__scene:
            intersection = obj.primitive.intersect(ray)
            if intersection and intersection[0] < hit_t:
                hit_object = obj
                hit_t, hit_point, hit_normal = intersection
        return hit_object, hit_point, hit_normal

    def __trace_diffuse(self, hit_object, hit_point, hit_normal):
        """Traces color of an object with diffuse material"""
        summed_color = Vector3()
        for light in filter(lambda obj: obj.is_light, self.__scene):
            transmission = Vector3(1, 1, 1)
            light_direction = (light.primitive.position - hit_point).normalize()
            for other in filter(lambda obj: obj != light, self.__scene):
                if other.primitive.intersect(Ray(hit_point + self.__bias * hit_normal,
                                             light_direction)):
                    transmission = Vector3()
                    break
            summed_color = summed_color + (
                hit_object.material.surface_color
                .mul_comp(transmission)
                .mul_comp(light.material.emission_color) *
                max(0, hit_normal.dot(light_direction)))
        return summed_color

    def __trace_non_diffuse(self, ray, hit_object, hit_point, hit_normal, depth):
        """Traces color of an object with refractive/reflective material"""
        inside = ray.direction.dot(hit_normal) > 0
        if inside:
            hit_normal = -hit_normal
        facing_ratio = -ray.direction.dot(hit_normal)
        fresnel = self.__mix((1 - facing_ratio) ** 2, 1, 0.1)
        reflection_ray = Ray(hit_point + self.__bias * hit_normal,
                             ray.direction.reflect(hit_normal).normalize())
        reflection = self.__trace_recursively(reflection_ray, depth + 1)
        refraction = Vector3()

        # transparent?
        if hit_object.material.transparency > 0:
            from_ior = ray.current_ior if inside else hit_object.material.ior
            to_ior = hit_object.material.ior if inside else ray.current_ior
            refraction_ray = Ray(hit_point - self.__bias * hit_normal,
                                 ray.direction.refract(from_ior, to_ior, hit_normal)
                                 .normalize())
            refraction = self.__trace_recursively(refraction_ray, depth + 1)

        # mix according to fresnel
        return ((reflection * fresnel +
                refraction * (1 - fresnel) * hit_object.material.transparency)
                .mul_comp(hit_object.material.surface_color))

    def __mix(self, a, b, mix):
        """Mixes to values by a factor"""
        return b * mix + a * (1 - mix)
