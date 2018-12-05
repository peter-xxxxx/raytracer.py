import os
import threading
from Queue import Queue
from tracer import Tracer
from tracer_gpu import Tracer_gpu
from vector3 import Vector3


class Renderer_gpu:
    """Renderer coordinating the tracing process"""

    def __init__(self, tilesize=32, threads=2):
        """Creates a new renderer"""
        self.__tilesize = tilesize
        self.__threads = threads

    def render(self, scene, camera, width, height, super_sampling=1, logging=True):
        """Renders a scene"""
        tracer = Tracer_gpu()
        self.__tiles = Queue()
        self.__rendered_tiles = Queue()
        self.__logging = logging

        # setup rendering threads
        if logging:
            print("render using {0} threads with {1}x{1} tiles".format(self.__threads, self.__tilesize))
        workers = []
        for i in range(self.__threads):
            thread = threading.Thread(target=self.__render_portion_gpu,
                                      args=(tracer, scene, camera, width, height, super_sampling))
            thread.start()
            workers.append(thread)

        # split image into tiles -> threads will begin computing
        self.__total_tiles = 0
        for y in range(0, height, self.__tilesize):
            for x in range(0, width, self.__tilesize):
                self.__tiles.put((x, y, x + self.__tilesize, y + self.__tilesize))
                self.__total_tiles += 1

        # wait and stop workers
        for i in range(self.__threads):
            self.__tiles.put(None)
        for thread in workers:
            thread.join()

        # merge results
        image = {}
        while not self.__rendered_tiles.empty():
            for tile in self.__rendered_tiles.get():
                for k, v in tile.items():
                    image[k] = v

        return image

    def __render_portion(self, tracer, scene, camera, width, height, super_sampling):
        """Worker method rendering a portion/tile of the image"""
        while True:
            next_work = self.__tiles.get()
            if next_work is None:
                break
            start_x, start_y, end_x, end_y = next_work
            rendered_tile = {}
            for y in range(start_y, min(end_y, height)):
                for x in range(start_x, min(end_x, width)):
                    sum_color = Vector3()
                    sampled_rays = 0
                    for ss_x in range(-super_sampling + 1, super_sampling):
                        for ss_y in range(-super_sampling + 1, super_sampling):
                            ray = camera.calcRay(x + ss_x, y + ss_y, width, height)
                            sum_color += tracer.trace(ray, scene)
                            sampled_rays += 1
                    rendered_tile[x, y] = sum_color * (1 / sampled_rays)
            self.__rendered_tiles.put((rendered_tile,))
            if self.__logging:
                finished = self.__rendered_tiles.qsize()
                print("{0}/{1} tiles | {2:.2f}%".format(finished, self.__total_tiles,
                                                        finished / self.__total_tiles * 100.0))
    def __render_portion_gpu(self, tracer, scene, camera, width, height, super_sampling):
        """Worker method rendering a portion/tile of the image"""
        while True:
            next_work = self.__tiles.get()
            if next_work is None:
                break
            start_x, start_y, end_x, end_y = next_work
            """
            : TODO: Implement with GPU
            """

            # Get Ray array

            # Tracer for Ray array with Scene

            rendered_tile = {}
            ray_array = []
            for y in range(start_y, min(end_y, height)):
                for x in range(start_x, min(end_x, width)):
                    sum_color = Vector3()
                    # sampled_rays = 0
                    ray = camera.calcRay(x, y, width, height)
                    ray_array.append([ray.origin, ray.direction, ray.current_ior])
                    ray_from_array.append([x, y])
                    # for ss_x in range(-super_sampling + 1, super_sampling):
                    #     for ss_y in range(-super_sampling + 1, super_sampling):
                    #         ray = camera.calcRay(x + ss_x, y + ss_y, width, height)
                    #         sum_color += tracer.trace(ray, scene)
                    #         sampled_rays += 1
                    # rendered_tile[x, y] = sum_color * (1 / sampled_rays)

            rendered_tile = tracer.trace(ray_array, ray_from_array, scene)
            """
            :# TODO:
            """


            self.__rendered_tiles.put((rendered_tile,))
            if self.__logging:
                finished = self.__rendered_tiles.qsize()
                print("{0}/{1} tiles | {2:.2f}%".format(finished, self.__total_tiles,
                                                        finished / self.__total_tiles * 100.0))
