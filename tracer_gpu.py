from ray import Ray
from vector3 import Vector3
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

"""
Ray: [ray.origin v3, ray.direction v3, ray.current_ior] len = 5

Object: [
         primitive_type=0,       [0]
         primitive_position, v3
         primitive_radius,       [4]

         material_surface_color, v3
         material_emission_color, v3
         material_reflectivity,
         material_transparency,
         material_ior,
         material_is_diffuse,
         is_light])

         len = 16

"""


class Tracer_gpu:
    """Main (ray) tracer coordinating the heavy algorithmic work"""

    def __init__(self, max_recursion_depth=5, bias=1e-4):
        """Creates a new tracer"""
        self.__max_recursion_depth = max_recursion_depth
        self.__bias = bias

        kernel_code = """
        #include "math.h"
        #include "float.h"

        __device__ void normalize_vector(float *a, float *b, float *c)
        {
            float len;
            float ta, tb, tc;

            ta = *a;
            tb = *b;
            tc = *c;
            len = ta*ta + tb*tb + tc*tc;

            if(len == 0.0)
                return;
            len = sqrt(len);
            *a = ta/len;
            *b = tb/len;
            *c = tc/len;
        }

        __device__ void intersectGPU(float *ray_array, float *scene, float *output, int *hit_obj_index, int scene_size)
        {

        /*
        input: ray_array, scene, scene_size
        output: [hit_point v3, hit_normal v3]
        */

            int ray_array_index = 0;
            int output_index;
            int i;
            int scene_index;
            int flag;
            int current_obj_i = -1;
            float l1, l2, l3;
            float t_ca;
            float d_squared;
            float radius_squared;
            float t_hc;
            float t;
            float hit_point1;
            float hit_point2;
            float hit_point3;
            float hit_normal1;
            float hit_normal2;
            float hit_normal3;
            float current_t = FLT_MAX;

            for(i = 0; i < scene_size; i++){
                flag = 0;
                scene_index = i*16
                if (scene[scene_index] < 0.5){ /* means sphere */
                    /* calculate intersect */
                    l1 = scene[scene_index+1] - ray_array[ray_array_index];
                    l2 = scene[scene_index+2] - ray_array[ray_array_index+1];
                    l3 = scene[scene_index+3] - ray_array[ray_array_index+2];
                    t_ca = l1*ray_array[ray_array_index+3] +
                           l2*ray_array[ray_array_index+4] +
                           l3*ray_array[ray_array_index+5];
                    if(t_ca < 0)
                        flag = 1;
                    d_squared = l1*l1 + l2*l2 +l3*l3 - t_ca*t_ca;
                    radius_squared = scene[scene_index+4]*scene[scene_index+4];
                    if(d_squared > radius_squared)
                        flag = 1;
                    t_hc = sqrt(radius_squared - d_squared)
                    t = t_ca - t_hc;
                    if(t < 0)
                        t = t_ca + t_hc;
                    hit_point1 = ray_array[ray_array_index] + t * ray_array[ray_array_index+3];
                    hit_point2 = ray_array[ray_array_index+1] + t * ray_array[ray_array_index+4];
                    hit_point3 = ray_array[ray_array_index+2] + t * ray_array[ray_array_index+5];
                    hit_normal1 = hit_point1 + scene[scene_index+1];
                    hit_normal2 = hit_point2 + scene[scene_index+2];
                    hit_normal3 = hit_point3 + scene[scene_index+3];
                    normalize_vector(&hit_normal1, &hit_normal2, &hit_normal3)

                    /* intersect python code
                    l = self.__position - ray.origin
                    t_ca = l.dot(ray.direction)
                    if t_ca < 0:
                        return
                    d_squared = l.dot(l) - t_ca ** 2
                    radius_squared = self.__radius ** 2
                    if d_squared > radius_squared:
                        return
                    t_hc = math.sqrt(radius_squared - d_squared)
                    t = t_ca - t_hc
                    if t < 0.0:
                        t = t_ca + t_hc
                    hit_point = ray.origin + t * ray.direction
                    hit_normal = (hit_point - self.__position).normalize()
                    return t, hit_point, hit_normal
                    */

                }
                if(flag == 0){
                    if(t < current_t){
                        current_t = t;
                        current_obj_i = i;
                        output[0] = hit_point1;
                        output[1] = hit_point2;
                        output[2] = hit_point3;
                        output[3] = hit_normal1;
                        output[4] = hit_normal2;
                        output[5] = hit_normal3;
                    }
                }
            }
            *hit_obj_index = current_obj_i;
        }

        __global__ void traceGPU(){

        }

        """



        """
        # TODO:
        """

    def trace(self, ray_array, scene):
        """Traces a ray through a scene to return the traced color"""
        self.__scene = scene
        self.__scene_gpu = self.__scene_to_gpu(scene)
        return self.__trace_gpu(ray_array, 0)

    def __scene_to_gpu(self, scene):
        scene_gpu = []
        for obj in scene:
            primitive = obj.primitive
            primitive_type = 0 #0
            primitive_position = primitive.position #123 v3
            primitive_radius = primitive.radius #4
            material = obj.material
            material_surface_color = material.surface_color #567 v3
            material_emission_color = material.emission_color #8910 v3
            material_reflectivity = material.reflectivity #11
            material_transparency = material.transparency #12
            material_ior = material.ior #13
            material_is_diffuse = material.is_diffuse #14
            is_light = obj.is_light #15

            scene_gpu.append([
                              primitive_type,
                              primitive_position,
                              primitive_radius,

                              material_surface_color,
                              material_emission_color,
                              material_reflectivity,
                              material_transparency,
                              material_ior,
                              material_is_diffuse,
                              is_light])

        return scene_gpu

    def __trace_gpu(ray_array):
        """
        : # TODO:

        recursive -> iterative

        every depth, call gpu func


        """

        while depth <= self.__max_recursion_depth:
            self.trace_func(ray_array_gpu, output_ray)


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
