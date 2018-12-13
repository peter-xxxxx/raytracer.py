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
        scene_index = i*16;
        if (scene[scene_index] < 0.5){ /* means sphere */
            /* calculate intersect */
            l1 = scene[scene_index+1] - ray_array[0];
            l2 = scene[scene_index+2] - ray_array[1];
            l3 = scene[scene_index+3] - ray_array[2];
            t_ca = l1*ray_array[3] +
                   l2*ray_array[4] +
                   l3*ray_array[5];
            if(t_ca < 0)
                flag = 1;
            d_squared = l1*l1 + l2*l2 +l3*l3 - t_ca*t_ca;
            radius_squared = scene[scene_index+4]*scene[scene_index+4];
            if(d_squared > radius_squared)
                flag = 1;
            t_hc = sqrt(radius_squared - d_squared);
            t = t_ca - t_hc;
            if(t < 0)
                t = t_ca + t_hc;
            hit_point1 = ray_array[0] + t * ray_array[3];
            hit_point2 = ray_array[1] + t * ray_array[4];
            hit_point3 = ray_array[2] + t * ray_array[5];
            hit_normal1 = hit_point1 - scene[scene_index+1];
            hit_normal2 = hit_point2 - scene[scene_index+2];
            hit_normal3 = hit_point3 - scene[scene_index+3];
            normalize_vector(&hit_normal1, &hit_normal2, &hit_normal3);
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

__device__ void trace_non_diffuseGPU (float *ray, int hit_obj_index, float *hit_point, float *hit_normal,
                                      float *scene, int scene_size, int depth, int max_depth, float *output);
__device__ void trace_diffuseGPU(int obj_index, float *hit_point, float *hit_normal, float *scene, int scene_size, float *output);

__device__ void trace_recursivelyGPU(float *ray, float *scene, int scene_size, float *output, int depth, int max_depth){
    /*
    hit_object, hit_point, hit_normal = self.__intersect(ray)
    if hit_object is None:
        return Vector3(0.3, 0.3, 0.3)  # horizon
    traced_color = Vector3()
    if not hit_object.material.is_diffuse and depth < self.__max_recursion_depth:
        traced_color = self.__trace_non_diffuse(ray, hit_object, hit_point, hit_normal, depth)
    else:
        traced_color = self.__trace_diffuse(hit_object, hit_point, hit_normal)
    return traced_color + hit_object.material.emission_color
    */

    int hit_obj_index;
    float output_intersect[6];

    intersectGPU(ray, scene, output_intersect, &hit_obj_index, scene_size);

    if (hit_obj_index < 0) {
        output[0] = 0.3;
        output[1] = 0.3;
        output[2] = 0.3;
        return;
    }

    float traced_color[3];

    if (depth < max_depth && scene[16*hit_obj_index+14] < 0.5)
        /* trace_non_diffuseGPU (ray, hit_obj_index, output_intersect, output_intersect + 3,
                              scene, scene_size, depth, max_depth, traced_color);*/
        trace_diffuseGPU (hit_obj_index, output_intersect, output_intersect + 3, scene, scene_size, traced_color);
    else
        trace_diffuseGPU (hit_obj_index, output_intersect, output_intersect + 3, scene, scene_size, traced_color);

    traced_color[0] += scene[16*hit_obj_index + 8];
    traced_color[1] += scene[16*hit_obj_index + 9];
    traced_color[2] += scene[16*hit_obj_index + 10];

    output[0] = traced_color[0];
    output[1] = traced_color[1];
    output[2] = traced_color[2];

}

__device__ void trace_diffuseGPU(int obj_index, float *hit_point, float *hit_normal, float *scene, int scene_size, float *output){
    /*
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
    */

    int i;
    float summed_color[3] = {0, 0, 0};
    float transmission[3] = {1, 1, 1};
    int hit_obj_index;

    float output_intersect[6];
    float current_ray[7];
    float temp;


    for(i = 0; i < scene_size; i++){
        if(scene[i*16 + 15] > 0.5){ /* is light */


            current_ray[0] = hit_point[0] + 0.0001*hit_normal[0];
            current_ray[1] = hit_point[1] + 0.0001*hit_normal[1];
            current_ray[2] = hit_point[2] + 0.0001*hit_normal[2];

            current_ray[3] = scene[i*16 + 1] - hit_point[0];
            current_ray[4] = scene[i*16 + 2] - hit_point[1];
            current_ray[5] = scene[i*16 + 3] - hit_point[2];

            current_ray[6] = 1.0;

            normalize_vector(current_ray+3, current_ray+4, current_ray+5);

            for(i = 0; i < scene_size; i++){
                if(scene[i*16 + 15] < 0.5)
                    intersectGPU(current_ray, scene+i, output_intersect, &hit_obj_index, 1);

                if(hit_obj_index < 0){
                    transmission[0] = 0;
                    transmission[1] = 0;
                    transmission[2] = 0;
                    break;
                }
            }

            temp = hit_normal[0]*current_ray[3] + hit_normal[1]*current_ray[4] + hit_normal[2]*current_ray[5];
            temp = 0 > temp ? 0 : temp;

            summed_color[0] += scene[obj_index*16 + 5] * transmission[0] * scene[i*16 + 8] * temp;
            summed_color[1] += scene[obj_index*16 + 6] * transmission[1] * scene[i*16 + 9] * temp;
            summed_color[2] += scene[obj_index*16 + 7] * transmission[2] * scene[i*16 + 10] * temp;
        }

    }
    output[0] = summed_color[0];
    output[1] = summed_color[1];
    output[2] = summed_color[2];

}

__device__ void trace_non_diffuseGPU (float *ray, int hit_obj_index, float *hit_point, float *hit_normal,
                                      float *scene, int scene_size, int depth, int max_depth, float *output)
{
    /*
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
    */

    if (ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2] > 0){
        hit_normal[0] *= -1;
        hit_normal[1] *= -1;
        hit_normal[2] *= -1;
    }
    float facing_ratio = - (ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2]);
    float fresnel = (1 - facing_ratio) * (1 - facing_ratio) * 0.9 + 0.1;

    float reflection_ray[7];

    reflection_ray[0] = hit_point[0] - 0.0001*hit_normal[0];
    reflection_ray[1] = hit_point[1] - 0.0001*hit_normal[1];
    reflection_ray[2] = hit_point[2] - 0.0001*hit_normal[2];

    float temp;
    temp = ray[0]*hit_normal[0] + ray[1]*hit_normal[1] + ray[2]*hit_normal[2];
    temp *= 2;

    reflection_ray[3] = ray[0] - temp * hit_normal[0];
    reflection_ray[4] = ray[1] - temp * hit_normal[1];
    reflection_ray[5] = ray[2] - temp * hit_normal[2];

    normalize_vector(reflection_ray+3, reflection_ray+4, reflection_ray+5);

    reflection_ray[6] = 1.0;

    float reflection[3];
    float refraction[3] = {0, 0, 0};

    trace_recursivelyGPU(reflection_ray, scene, scene_size, reflection, depth+1, max_depth);

    /*
    def refract(self, from_ior, to_ior, normal):
        # Refracts the vector with regard to material change and normal

        eta = to_ior / from_ior
        cos_i = -normal.dot(self)
        k = 1 - eta ** 2 * (1 - cos_i ** 2)
        return self * eta + normal * (eta * cos_i - math.sqrt(k))
    */

    if (scene[hit_obj_index*16 + 12] > 0){
        float from_ior, to_ior;
        if (ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2] > 0) {
            from_ior = ray[6];
            to_ior = scene[hit_obj_index*16 + 13];
        }
        else {
            from_ior = scene[hit_obj_index*16 + 13];
            to_ior = ray[6];
        }

        float refraction_ray[7];

        refraction_ray[0] = hit_point[0] - 0.0001*hit_normal[0];
        refraction_ray[1] = hit_point[1] - 0.0001*hit_normal[1];
        refraction_ray[2] = hit_point[2] - 0.0001*hit_normal[2];

        float eta = to_ior / from_ior;
        float cos_i = -(ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2]);
        float k = 1 - eta*eta * (1 - cos_i*cos_i);

        temp = eta*cos_i - sqrt(k);

        refraction_ray[3] = ray[3] * eta + hit_normal[0] * temp;
        refraction_ray[4] = ray[4] * eta + hit_normal[1] * temp;
        refraction_ray[5] = ray[5] * eta + hit_normal[2] * temp;

        normalize_vector(refraction_ray+3, refraction_ray+4, refraction_ray+5);

        refraction_ray[6] = 1;



        trace_recursivelyGPU(refraction_ray, scene, scene_size, refraction, depth+1, max_depth);

    }

    output[0] = (reflection[0] * fresnel + refraction[0] * (1-fresnel) * scene[hit_obj_index*16 + 12]) * scene[hit_obj_index*16 + 5];
    output[1] = (reflection[1] * fresnel + refraction[1] * (1-fresnel) * scene[hit_obj_index*16 + 12]) * scene[hit_obj_index*16 + 6];
    output[2] = (reflection[2] * fresnel + refraction[2] * (1-fresnel) * scene[hit_obj_index*16 + 12]) * scene[hit_obj_index*16 + 7];



}


__global__ void traceGPU(float *ray_array, float *scene, int width, int scene_size, float *output, int max_depth){

    int tx;
    int ray_array_index;

    tx = blockIdx.x*blockDim.x + threadIdx.x;


    if (tx >= width)
        return;

    ray_array_index = tx;

    int hit_obj_index;
    float output_c[3];



    trace_recursivelyGPU(ray_array + ray_array_index*7, scene, scene_size, output_c, 0, max_depth);

    output[0] = output_c[0];
    output[1] = output_c[1];
    output[2] = output_c[2];

}
