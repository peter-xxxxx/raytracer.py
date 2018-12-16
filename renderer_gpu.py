import os
import threading
from Queue import Queue
from tracer import Tracer
from tracer_gpu import Tracer_gpu
from vector3 import Vector3


class Renderer_gpu:
    """
    : Renderer coordinating the tracing process
    : GPU version
    """

    def __init__(self, tilesize=32, threads=2):
        """Creates a new renderer"""
        self.__tilesize = tilesize
        self.__threads = threads

    def render(self, scene, camera, width, height, super_sampling=1, logging=True):
        """Renders a scene"""
        tracer = Tracer_gpu()

        ray_array = []
        ray_from_array = []
        # generate ray array
        for y in range(0, height):
            for x in range(0, width):
                sum_color = Vector3()
                # calculate ray using Camera class
                ray = camera.calcRay(x, y, width, height)
                # put ray into an array for further processing
                ray_array.append([ray.origin.x, ray.origin.y, ray.origin.z,
                                  ray.direction.x, ray.direction.y, ray.direction.z, ray.current_ior])
                ray_from_array.append([x, y])
        # call GPU tracer
        rendered = tracer.trace(ray_array, ray_from_array, scene)

        return rendered
