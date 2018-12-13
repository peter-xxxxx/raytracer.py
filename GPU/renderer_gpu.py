import os
import threading
from Queue import Queue
from tracer_gpu import Tracer
from vector3 import Vector3
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

class Renderer:
    """Renderer coordinating the tracing process"""

    def __init__(self, tilesize=32, threads=2):
        """Creates a new renderer"""
        self.__tilesize = tilesize
        self.__threads = threads

    def render_gpu(self, scene, camera, width, height, channel = 3, super_sampling=1, logging=True):
        """Renders a scene"""

        # data on GPU
        img_gpu = gpuarray.empty((width, height, channel), np.float32)
        scene_gpu_test = gpuarray.empty((width, height, channel), np.float32)
        camera_gpu_test = gpuarray.empty(1, np.float32)

        # parallel parameters
        threadsPerBlock = 2 * super_sampling - 1
        blocksPerGrid = width * height
        nThreads = blocksPerGrid * threadsPerBlock

        # Kernel Code
        # NO super sampling
        kernel_render = """
        #include <stdio.h>

        struct Vector3{
            float x = 0, y = 0, z = 0;
        }

        __global__ void RenderFunc(float* img, float* scene, float* camera){
            int cordx = blockIdx.x;
            int cordy = blockIdx.y;
            int pos = cordx * cordy * 3 + 1;
            Vector3 pixel;

            //ray trace

            img[pos] = pixel.x;
            img[pos + 1] = pixel.y
            img[pos + 2] = pixel.z;
        }
        """

        # Compile Source Code
        mod = compiler.SourceModule(kernel_render)
        RenderFunc = mod.get_function("RenderFunc")

        RenderFunc(img_gpu, scene_gpu_test, camera_gpu_test,
                block = (threadsPerBlock, 1, 1),
                grid  = (np.int32(width), np.int32(height), 1))

        return img_gpu.get()
