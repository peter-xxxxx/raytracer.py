class Ray:
    """Ray (/half-line) class"""

    def __init__(self, origin, direction, ior=1.0):
        """Creates a new ray"""
        self.__origin = origin
        self.__direction = direction
        self.__current_ior = ior

    @property
    def origin(self):
        """Returns the origin of the ray"""
        return self.__origin

    @property
    def direction(self):
        """Returns the direction of the ray"""
        return self.__direction

    @property
    def current_ior(self):
        """Returns the current index of refraction of the ray"""
        return self.__current_ior

    @property
    def to_list(self):
        return [self.__origin, self.__direction, self.__current_ior]
