# raytracer.py - basic Python raytracer
# Micha Hanselmann, 2017
# Peter Xu, 2018
# based on http://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-ray-tracing

from camera import Camera
from material import Material
from sphere import Sphere
from renderer_gpu import Renderer_gpu
from renderobject import RenderObject
from vector3 import Vector3
import time

# render settings
width = 640
height = 480
super_sampling = 1

"""
create demo scene
"""

# RGB = (255, 255, 255) - White
light_mat = Material(emission_color=Vector3(4, 4, 4))
light_sphere = Sphere(Vector3(5, 50, 20), 5)
light_obj = RenderObject(light_sphere, light_mat)

# RGB = (255, 255, 255) - White
light2_mat = Material(emission_color=Vector3(2, 2, 2))
light2_sphere = Sphere(Vector3(0, 10, -5), 5)
light2_obj = RenderObject(light2_sphere, light2_mat)

# RGB = (0, 0, 255) - Blue
blue_light_mat = Material(emission_color=Vector3(0, 0, 1))
blue_light_sphere = Sphere(Vector3(-40, 5, 10), 3)
blue_light_obj = RenderObject(blue_light_sphere, blue_light_mat)

# GGB = (51, 51, 51) - Deep Grey
ground_mat = Material(surface_color=Vector3(0.2, 0.2, 0.2))
ground_sphere = Sphere(Vector3(0, -10010, 20), 10000)
ground_obj = RenderObject(ground_sphere, ground_mat)

# RGB = (204, 51, 51) - Deep Red
mat1 = Material(surface_color=Vector3(0.8, 0.2, 0.2))
sph1 = Sphere(Vector3(10, 1, 50), 5)
obj1 = RenderObject(sph1, mat1)

# RGB = (255, 255, 255) - Black
mat2 = Material(surface_color=Vector3(1, 1, 1), transparency=1, ior=1.1)
sph2 = Sphere(Vector3(1, -1, 10), 1)
obj2 = RenderObject(sph2, mat2)

# RGB = (204, 204, 255) - Purple
mat3 = Material(surface_color=Vector3(0.8, 0.8, 1), reflectivity=0.5, transparency=0.8)
sph3 = Sphere(Vector3(-4, -1, 20), 2)
obj3 = RenderObject(sph3, mat3)

# RGB = (255, 255, 255) - Black
mat4 = Material(surface_color=Vector3(1, 1, 1), reflectivity=1.0)
sph4 = Sphere(Vector3(0, 0, 60), 8)
obj4 = RenderObject(sph4, mat4)

scene = [ground_obj, light_obj, light2_obj, blue_light_obj, obj1, obj2, obj3, obj4]

# render
renderer = Renderer_gpu()
camera = Camera(Vector3(), 30)

start = time.time()
rendered = renderer.render(scene, camera, width, height)
end = time.time()

print("cost ", end-start, " time to run")

image = rendered

# write image file
def to_rgb_color(x, y, image):
        """Converts the vector into RGB values"""
        r = max(0, min(1, image[x+y*width, 0])) * 255
        g = max(0, min(1, image[x+y*width, 1])) * 255
        b = max(0, min(1, image[x+y*width, 2])) * 255
        return int(r), int(g), int(b)

file = open("output_gpu.ppm", "w")
file.write("P3\n{0} {1}\n255\n".format(width, height))
for y in range(height):
    for x in range(width):
        file.write("{0} {1} {2} ".format(*to_rgb_color(x,y,image)))
file.close()
