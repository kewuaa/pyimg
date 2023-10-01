import sys
from distutils.unixccompiler import UnixCCompiler
from os import environ

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import numpy as np

include_dirs = [
    environ['include'],
    np.get_include()
]
library_dirs = [
    environ['lib']
]


class build(build_ext):
    def build_extensions(self):
        if isinstance(self.compiler, UnixCCompiler):
            if 'zig' in self.compiler.cc:
                self.compiler.dll_libraries.clear()
                self.compiler.set_executable(
                    'compiler_so',
                    f'{self.compiler.cc} -O3 -Wall'
                )
                for ext in self.extensions:
                    ext.undef_macros.append("_DEBUG")
                    if "-fopenmp" in ext.extra_compile_args:
                        ext.libraries.append("libomp")
        for ext in self.extensions:
            ext.include_dirs = include_dirs
            ext.library_dirs = library_dirs
        super().build_extensions()


if "--use-cython" in sys.argv:
    sys.argv.remove("--use-cython")
    use_cython = True
else:
    use_cython = False
suffix = "pyx" if use_cython else "c"
exts = [
    Extension(
        name='pyimg.core',
        sources=['src\\pyimg\\core.' + suffix],
    ),
    Extension(
        name='pyimg.fft',
        sources=['src\\pyimg\\fft.' + suffix],
    ),
    Extension(
        name='pyimg.utils',
        sources=['src\\pyimg\\utils.' + suffix],
    )
]
if use_cython:
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    cmdclass={'build_ext': build},
)
