import math
from ray import Ray
from vector3 import Vector3


class Camera:
    """Camera in a scene"""

    def __init__(self, position, fov):
        """Creates a new camera"""
        self.__position = position
        self.__fov = fov # Field of View

    # every ray starts at the original point of the camera, goes
    # to the project image plane
    def calcRay(self, x, y, width, height):
        """Calculates the ray (to be traced) at image position"""
        aspect_ratio = width / height
        angle = math.tan(math.pi * 0.5 * self.__fov / 180)
        x_norm = (2 * ((x + 0.5) / width) - 1) * angle * aspect_ratio
        y_norm = (1 - 2 * ((y + 0.5) / height)) * angle
        look_at = Vector3(x_norm, y_norm, 1).normalize()
        return Ray(self.__position, look_at)
