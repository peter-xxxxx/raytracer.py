from ray import Ray
from vector3 import Vector3
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

"""
Ray: [ray.origin v3, ray.direction v3, ray.current_ior] len = 7

Object: [
         primitive_type=0,       [0]
         primitive_position, v3
         primitive_radius,       [4]

         material_surface_color, v3 [5-7]
         material_emission_color, v3 [8-10]
         material_reflectivity, [11]
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

        kernel_code_intersect = """
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
                if (true){ /* means sphere */
                    /* calculate intersect */
                    l1 = scene[scene_index+1] - ray_array[0];
                    l2 = scene[scene_index+2] - ray_array[1];
                    l3 = scene[scene_index+3] - ray_array[2];
                    t_ca = l1*ray_array[3] +
                           l2*ray_array[4] +
                           l3*ray_array[5];
                    if(t_ca < 0)
                        flag = 1;
                    d_squared = l1*l1 + l2*l2 + l3*l3 - t_ca*t_ca;
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

        __global__ void intersectKernel(float *ray_array, float *scene, int width, int scene_size, float *output, int *output_obj_index){

            int tx;
            int ray_array_index;

            tx = blockIdx.x*blockDim.x + threadIdx.x;


            if (tx >= width)
                return;

            ray_array_index = tx * 7;

            int hit_obj_index;
            float intersect_output[6];

            intersectGPU(ray_array+ray_array_index, scene, intersect_output, &hit_obj_index, scene_size);

            output[0 + 6*tx] = intersect_output[0];
            output[1 + 6*tx] = intersect_output[1];
            output[2 + 6*tx] = intersect_output[2];
            output[3 + 6*tx] = intersect_output[3];
            output[4 + 6*tx] = intersect_output[4];
            output[5 + 6*tx] = intersect_output[5];

            output_obj_index[tx] = hit_obj_index;

        }


        """

        self.mod = compiler.SourceModule(kernel_code_intersect)

        self.kernel_fun_intersect = self.mod.get_function("intersectKernel")

        kernel_code_trace_diffuse = """
        #include "math.h"
        #include "float.h"
        #include "stdio.h"

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
            *a = ta*(1/len);
            *b = tb*(1/len);
            *c = tc*(1/len);
        }

        __device__ bool hitobjGPU(float *ray, float *obj){
            float l1, l2, l3;
            float t_ca;
            float d_squared;
            float radius_squared;

            l1 = obj[1] - ray[0];
            l2 = obj[2] - ray[1];
            l3 = obj[3] - ray[2];
            t_ca = l1*ray[3] +
                   l2*ray[4] +
                   l3*ray[5];
            if(t_ca < 0)
                return false;
            d_squared = l1*l1 + l2*l2 +l3*l3 - t_ca*t_ca;
            radius_squared = obj[4]*obj[4];
            if(d_squared > radius_squared)
                return false;
            return true;
        }

        __device__ void trace_diffuseGPU(int obj_index, float *hit_point, float *hit_normal, float *scene, int scene_size, float *output){
            int i;
            int j;
            float summed_color[3] = {0, 0, 0};

            int hit_obj_index;

            float output_intersect[6];
            float current_ray[7];
            float temp;
            float transmission[3];

            for(i = 0; i < scene_size; i++){
                if(scene[i*16 + 15] > 0.5){ /* is light */
                    transmission[0] = 1;
                    transmission[1] = 1;
                    transmission[2] = 1;
                    current_ray[0] = hit_point[0] + 0.0001*hit_normal[0];
                    current_ray[1] = hit_point[1] + 0.0001*hit_normal[1];
                    current_ray[2] = hit_point[2] + 0.0001*hit_normal[2];
                    current_ray[3] = scene[i*16 + 1] - hit_point[0];
                    current_ray[4] = scene[i*16 + 2] - hit_point[1];
                    current_ray[5] = scene[i*16 + 3] - hit_point[2];
                    current_ray[6] = 1.0;
                    normalize_vector(current_ray+3, current_ray+4, current_ray+5);
                    for(j = 0; j < scene_size; j++){
                        if (j != i) {
                            if(hitobjGPU(current_ray, scene + j*16)){
                                transmission[0] = 0;
                                transmission[1] = 0;
                                transmission[2] = 0;
                                break;
                            }
                        }
                    }
                    temp = hit_normal[0]*current_ray[3] + hit_normal[1]*current_ray[4] + hit_normal[2]*current_ray[5];
                    temp = 0 > temp ? 0 : temp;

                    summed_color[0] += scene[obj_index*16 + 5] * transmission[0] * scene[i*16 + 8] * temp;
                    summed_color[1] += scene[obj_index*16 + 6] * transmission[1] * scene[i*16 + 9] * temp;
                    summed_color[2] += scene[obj_index*16 + 7] * transmission[2] * scene[i*16 + 10] * temp;
                }
            }
            output[0] = summed_color[0] + scene[obj_index*16 + 8];
            output[1] = summed_color[1] + scene[obj_index*16 + 9];
            output[2] = summed_color[2] + scene[obj_index*16 + 10];

        }

        __global__ void traceDiffuseKernel(float *intersect,
                                           int *hit_obj_array, int width,
                                           float *scene, int scene_size, int flag,
                                           float *output){

            int tx;
            int ray_array_index;
            int hit_obj_index;

            tx = blockIdx.x*blockDim.x + threadIdx.x;
            hit_obj_index = hit_obj_array[tx];

            if (tx >= width)
                return;

            if (hit_obj_index < 0) {
                output[tx*3 + 0] = 0.3;
                output[tx*3 + 1] = 0.3;
                output[tx*3 + 2] = 0.3;
                return;
            }

            if (scene[16*hit_obj_index+14] < 0.5 && flag == 0){
                output[tx*3 + 0] = 0.0;
                output[tx*3 + 1] = 0.0;
                output[tx*3 + 2] = 0.0;
                return;
            }

            ray_array_index = tx * 7;

            float output_c[3];

            trace_diffuseGPU(hit_obj_index, intersect+tx*6, intersect+tx*6+3, scene, scene_size, output_c);

            output[tx*3 + 0] = output_c[0];
            output[tx*3 + 1] = output_c[1];
            output[tx*3 + 2] = output_c[2];

        }


        """

        self.mod = compiler.SourceModule(kernel_code_trace_diffuse)

        self.kernel_fun_trace_diffuse = self.mod.get_function("traceDiffuseKernel")

        kernel_code_trace_non_diffuse = """
        #include "math.h"
        #include "float.h"
        #include "stdio.h"

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
            *a = ta*(1/len);
            *b = tb*(1/len);
            *c = tc*(1/len);
        }

        __device__ void trace_non_diffuseGPU (float *ray, int hit_obj_index, float *hit_point, float *hit_normal_origin,
                                              float *scene, int scene_size, float *reflect_ray, float *refract_ray,
                                              float *fresnel)
        {
            float hit_normal[3];

            if (ray[3]*hit_normal_origin[0] + ray[4]*hit_normal_origin[1] + ray[5]*hit_normal_origin[2] > 0){
                hit_normal[0] = -hit_normal_origin[0];
                hit_normal[1] = -hit_normal_origin[1];
                hit_normal[2] = -hit_normal_origin[2];
            }
            else{
                hit_normal[0] = hit_normal_origin[0];
                hit_normal[1] = hit_normal_origin[1];
                hit_normal[2] = hit_normal_origin[2];
            }
            float facing_ratio = -(ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2]);
            *fresnel = (1 - facing_ratio) * (1 - facing_ratio) * 0.9 + 0.1;

            reflect_ray[0] = hit_point[0] - 0.0001*hit_normal[0];
            reflect_ray[1] = hit_point[1] - 0.0001*hit_normal[1];
            reflect_ray[2] = hit_point[2] - 0.0001*hit_normal[2];

            float temp;
            temp = ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2];
            temp *= 2;

            reflect_ray[3] = ray[3] - temp * hit_normal[0];
            reflect_ray[4] = ray[4] - temp * hit_normal[1];
            reflect_ray[5] = ray[5] - temp * hit_normal[2];

            normalize_vector(reflect_ray+3, reflect_ray+4, reflect_ray+5);

            reflect_ray[6] = 1.0;

            refract_ray[6] = 0.0;

            if (scene[hit_obj_index*16 + 12] > 0){
                float from_ior, to_ior;
                if (ray[3]*hit_normal_origin[0] + ray[4]*hit_normal_origin[1] + ray[5]*hit_normal_origin[2] > 0) {
                    from_ior = ray[6];
                    to_ior = scene[hit_obj_index*16 + 13];
                }
                else {
                    from_ior = scene[hit_obj_index*16 + 13];
                    to_ior = ray[6];
                }


                refract_ray[0] = hit_point[0] - 0.0001*hit_normal[0];
                refract_ray[1] = hit_point[1] - 0.0001*hit_normal[1];
                refract_ray[2] = hit_point[2] - 0.0001*hit_normal[2];

                float eta = to_ior / from_ior;
                float cos_i = -(ray[3]*hit_normal[0] + ray[4]*hit_normal[1] + ray[5]*hit_normal[2]);
                float k = 1 - eta*eta * (1 - cos_i*cos_i);

                temp = eta*cos_i - sqrt(k);

                refract_ray[3] = ray[3] * eta + hit_normal[0] * temp;
                refract_ray[4] = ray[4] * eta + hit_normal[1] * temp;
                refract_ray[5] = ray[5] * eta + hit_normal[2] * temp;

                normalize_vector(refract_ray+3, refract_ray+4, refract_ray+5);

                refract_ray[6] = 1.0;

            }

        }

        __global__ void traceNonDiffuseKernel(float *ray_array, float *intersect,
                                              int *hit_obj_array, int width,
                                              float *scene, int scene_size,
                                              float *reflect_ray, float *refract_ray,
                                              float *fresnel){

            int tx;
            int ray_array_index;
            int hit_obj_index;

            tx = blockIdx.x*blockDim.x + threadIdx.x;
            hit_obj_index = hit_obj_array[tx];

            ray_array_index = tx * 7;

            if (tx >= width) {
                return;
            }

            if (hit_obj_index < 0 || scene[16*hit_obj_index + 14] > 0.5) {
                reflect_ray[ray_array_index + 6] = 0;
                refract_ray[ray_array_index + 6] = 0;
                return;
            }

            trace_non_diffuseGPU (ray_array + ray_array_index, hit_obj_index, intersect + tx*6, intersect + tx*6 + 3,
                                  scene, scene_size, reflect_ray + ray_array_index, refract_ray + ray_array_index,
                                  fresnel + tx);

        }


        """

        self.mod = compiler.SourceModule(kernel_code_trace_non_diffuse)

        self.kernel_fun_trace_non_diffuse = self.mod.get_function("traceNonDiffuseKernel")


        kernel_code_process_stack = """
        #include "math.h"
        #include "float.h"
        #include "stdio.h"

        __global__ void processStackKernel(int *ray_index,
                                           int width,
                                           float *scene,
                                           float *color,
                                           float *fresnel,
                                           int *hit_obj_array,
                                           float *output_color){

            int tx;
            int output_index;
            int hit_obj_index;

            tx = blockIdx.x*blockDim.x + threadIdx.x;
            hit_obj_index = hit_obj_array[tx];

            output_index = ray_index[tx];

            if (tx >= width) {
                return;
            }

            if (tx > 0 && ray_index[tx-1] == ray_index[tx]) {
                return;
            }

            float refraction[3] = {0, 0, 0};
            if (tx+1 < width && ray_index[tx] == ray_index[tx+1]){
                refraction[0] = color[tx*3 + 3];
                refraction[1] = color[tx*3 + 4];
                refraction[2] = color[tx*3 + 5];
            }

            output_color[output_index*3 + 0] =
                (color[tx*3 + 0] * fresnel[tx] +
                refraction[0] * (1 - fresnel[tx]) * scene[hit_obj_index*16 + 12]) *
                scene[hit_obj_index*16 + 5] + scene[hit_obj_index*16 + 8];
            output_color[output_index*3 + 1] =
                (color[tx*3 + 1] * fresnel[tx] +
                refraction[1] * (1 - fresnel[tx]) * scene[hit_obj_index*16 + 12]) *
                scene[hit_obj_index*16 + 6] + scene[hit_obj_index*16 + 9];
            output_color[output_index*3 + 2] =
                (color[tx*3 + 2] * fresnel[tx] +
                refraction[2] * (1 - fresnel[tx]) * scene[hit_obj_index*16 + 12]) *
                scene[hit_obj_index*16 + 7] + scene[hit_obj_index*16 + 10];

        }


        """

        self.mod = compiler.SourceModule(kernel_code_process_stack)

        self.kernel_fun_process_stack = self.mod.get_function("processStackKernel")


    def trace(self, ray_array, ray_from_array, scene):
        """Traces a ray through a scene to return the traced color"""
        self.__scene = scene
        self.__ray_from_array = ray_from_array
        self.__scene_gpu = self.scene_to_gpu(scene)

        return self.trace_gpu(ray_array)

    def scene_to_gpu(self, scene):
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
                              primitive_position.x,
                              primitive_position.y,
                              primitive_position.z,
                              primitive_radius,

                              material_surface_color.x,
                              material_surface_color.y,
                              material_surface_color.z,
                              material_emission_color.x,
                              material_emission_color.y,
                              material_emission_color.z,
                              material_reflectivity,
                              material_transparency,
                              material_ior,
                              material_is_diffuse,
                              is_light])
                              # 0, Vector3(-4, -1, 20), 2,
                              # Vector3(0.8, 0.8, 1), Vector3(0, 0, 0), 0.0, 0.8, 1.0, False, False
        return scene_gpu

    def trace_gpu(self, ray_array):
        """
        : # TODO:

        recursive -> iterative

        every depth, call gpu func


        """

        stack = []

        ray_array_cpu = np.array(ray_array, np.float32)
        scene_cpu = np.array(self.__scene_gpu, np.float32)
        print ray_array_cpu.shape
        print scene_cpu.shape

        scene_size = np.int32(scene_cpu.shape[0])


        for i in range(self.__max_recursion_depth):
            num_ray = np.int32(ray_array_cpu.shape[0])

            num_thread = 256
            num_block = int(np.ceil(float(num_ray)/float(num_thread)))
            ray_array_gpu = gpuarray.to_gpu(ray_array_cpu)
            scene_gpu = gpuarray.to_gpu(scene_cpu)

            output_intersect_gpu = gpuarray.empty((num_ray, 6), np.float32)
            output_obj_index_gpu = gpuarray.empty((num_ray, 1), np.int32)
            output_color = gpuarray.empty((num_ray, 3), np.float32)

            self.kernel_fun_intersect(ray_array_gpu, scene_gpu, num_ray, scene_size, output_intersect_gpu, output_obj_index_gpu,
                                      block = (num_thread, 1, 1),
                                      grid = (num_block, 1, 1))

            intersect_cpu = output_intersect_gpu.get()
            obj_index_cpu = output_obj_index_gpu.get()

            intersect_gpu = gpuarray.to_gpu(intersect_cpu)
            obj_index_gpu = gpuarray.to_gpu(obj_index_cpu)

            if i == self.__max_recursion_depth-1:
                flag = np.int32(1)
            else:
                flag = np.int32(0)

            self.kernel_fun_trace_diffuse(intersect_gpu,
                                          obj_index_gpu,
                                          num_ray, scene_gpu, scene_size, flag,
                                          output_color,
                                          block = (num_thread, 1, 1),
                                          grid = (num_block, 1, 1))

            color_cpu = output_color.get()

            if flag == 1:
                stack.append([color_cpu])
                break

            intersect_gpu = gpuarray.to_gpu(intersect_cpu)
            obj_index_gpu = gpuarray.to_gpu(obj_index_cpu)
            output_reflect_ray_gpu = gpuarray.empty((num_ray, 7), np.float32)
            output_refract_ray_gpu = gpuarray.empty((num_ray, 7), np.float32)
            output_fresnel_gpu = gpuarray.empty((num_ray, 1), np.float32)

            self.kernel_fun_trace_non_diffuse(ray_array_gpu,
                                              intersect_gpu,
                                              obj_index_gpu,
                                              num_ray, scene_gpu, scene_size, flag,
                                              output_reflect_ray_gpu,
                                              output_refract_ray_gpu,
                                              output_fresnel_gpu,
                                              block = (num_thread, 1, 1),
                                              grid = (num_block, 1, 1))

            reflect_ray_cpu = output_reflect_ray_gpu.get()
            refract_ray_cpu = output_refract_ray_gpu.get()
            fresnel_cpu = output_fresnel_gpu.get()

            ray_array_cpu_next, to_stack = self.ray_batch_filter(reflect_ray_cpu,
                                                                 refract_ray_cpu,
                                                                 fresnel_cpu,
                                                                 obj_index_cpu)

            to_stack.append(color_cpu)

            stack.append(to_stack)

            ray_array_cpu = np.array(ray_array_cpu_next, np.float32)


        color = self.process_stack(stack, scene_gpu)

        return color

    def process_stack(self, stack, scene_gpu):
        num_thread = 256
        color_cpu = np.array(stack[len(stack)-1][0], np.float32)

        for i in range(len(stack)-1, 0, -1):
            width = np.int32(len(stack[i-1][0]))
            num_block = int(np.ceil(float(width)/float(num_thread)))
            ray_index_cpu = np.array(stack[i-1][0], np.int32)
            fresnel_cpu = np.array(stack[i-1][1], np.float32)
            obj_index_cpu = np.array(stack[i-1][2], np.int32)
            output_color_cpu = np.array(stack[i-1][3], np.float32)

            color_gpu = gpuarray.to_gpu(color_cpu)
            ray_index_gpu = gpuarray.to_gpu(ray_index_cpu)
            fresnel_gpu = gpuarray.to_gpu(fresnel_cpu)
            obj_index_gpu = gpuarray.to_gpu(obj_index_cpu)
            output_color_gpu = gpuarray.to_gpu(output_color_cpu)

            self.kernel_fun_process_stack(ray_index_gpu,
                                          width,
                                          scene_gpu,
                                          color_gpu,
                                          fresnel_gpu,
                                          obj_index_gpu,
                                          output_color_gpu,
                                          block = (num_thread, 1, 1),
                                          grid = (num_block, 1, 1))

            output_color_cpu = output_color_gpu.get()
            color_cpu = output_color_cpu


        return color_cpu



    def ray_batch_filter(self, reflect_ray, refract_ray, fresnel, obj_index):
        index_n = []
        ray_n = []
        fresnel_n = []
        obj_index_n = []
        for i in range(len(reflect_ray)):
            if reflect_ray[i][6] > 0.5:
                ray_n.append(reflect_ray[i])
                index_n.append(i)
                fresnel_n.append(fresnel[i])
                obj_index_n.append(obj_index[i])
                if refract_ray[i][6] > 0.5:
                    ray_n.append(refract_ray[i])
                    index_n.append(i)
                    fresnel_n.append(fresnel[i])
                    obj_index_n.append(obj_index[i])

        return ray_n, [index_n, fresnel_n, obj_index_n]


    
