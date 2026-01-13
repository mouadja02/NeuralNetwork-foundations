from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cmatrix",
        ["c/bindings/matrix_pybind.cpp", "c/src/matrix.c"],
        include_dirs=["c/include"],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="cmatrix",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)