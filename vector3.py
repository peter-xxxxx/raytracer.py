import math


class Vector3:
    """3-dim vector class"""

    def __init__(self, x=0, y=0, z=0):
        """Creates a new vector"""
        self.__x = x
        self.__y = y
        self.__z = z

    def __str__(self):
        """Returns a string representation of the vector"""
        return "Vector3({0}, {1}, {2})".format(self.__x, self.__y, self.__z)

    __repr__ = __str__

    def __add__(self, other):
        """Adds another vector to this"""
        return Vector3(self.__x + other.__x,
                       self.__y + other.__y,
                       self.__z + other.__z)

    def __sub__(self, other):
        """Subtracts another vector to this"""
        return self + (-other)

    def __mul__(self, factor):
        """Multiplies this vector by a factor"""
        return Vector3(self.__x * factor,
                       self.__y * factor,
                       self.__z * factor)

    __rmul__ = __mul__

    def __neg__(self):
        """Negates the vector"""
        return Vector3(-self.__x, -self.__y, -self.__z)

    @property
    def length(self):
        """Returns the length of the vector"""
        return math.sqrt(self.__x ** 2 + self.__y ** 2 + self.__z ** 2)

    def normalize(self):
        """Normalizes the vector"""
        length = self.length
        if length == 0.0:
            return self
        return (1.0 / length) * self

    def dot(self, other):
        """Dot-product of another vector and this"""
        return (self.__x * other.__x +
                self.__y * other.__y +
                self.__z * other.__z)

    def to_rgb_color(self):
        """Converts the vector into RGB values"""
        r = max(0, min(1, self.__x)) * 255
        g = max(0, min(1, self.__y)) * 255
        b = max(0, min(1, self.__z)) * 255
        return int(r), int(g), int(b)

    def mul_comp(self, other):
        """Multiplies another vector with this component-wise"""
        return Vector3(self.__x * other.__x,
                       self.__y * other.__y,
                       self.__z * other.__z)

    def reflect(self, normal):
        """Reflects the vector at the plane represented by the normal"""
        return self - 2.0 * normal.dot(self) * normal

    def refract(self, from_ior, to_ior, normal):
        """Refracts the vector with regard to material change and normal"""
        eta = to_ior / from_ior
        cos_i = -normal.dot(self)
        k = 1 - eta ** 2 * (1 - cos_i ** 2)
        return self * eta + normal * (eta * cos_i - math.sqrt(k))
