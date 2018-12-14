## raytracer.py

This is a very easy understandable Python version ray tracing program code by DeerMichel.

We used the PyCUDA to accelerate the ray tracing rendering program as our heterogeneous computing course. Now it can finish rendering in minute. Works on tesseract server, too.

Prefer to run on Python 2, need to change `Queue` to `queue` to run on Python 3

`python main.py` to run CPU program

`python main_gpu.py` to run CUDA GPU version program

The output is a PPM format image file.

> From orginal arthor:

> A basic raytracer that will render you a fancy spherific demo scene via
`python3 main.py`. Guess what, it's slow! One could speed things up by using
numpy and doing other nifty optimizations... However this project was meant to
get to know Python - not to create a state-of-the-art-super-pathtracer - instead just
use C++ and be fine. Or check out [Minilight](http://www.hxa.name/minilight/) -
a beautiful GI path tracer (keep in mind: Python won't get faster ^^).
Nevertheless, if you want to improve sth (e.g. fix the super-sampling)... feel
free to do so :sweat_smile:.

![Rendered Demo Scene](_demo.png)
