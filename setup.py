from setuptools import Extension
from setuptools import setup

setup(
    name="laser_polio",
    ext_modules=[
        Extension(
            "laser_polio.compiled",
            sources=["src/laser_polio/compiled.cpp"],
            extra_compile_args=["-Xclang", "-fopenmp", "-I/opt/homebrew/opt/libomp/include"],
            extra_link_args=["-lomp", "-L/opt/homebrew/opt/libomp/lib"],
        ),
    ],
)
