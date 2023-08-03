import sys
from distutils.unixccompiler import UnixCCompiler
from os import environ

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
                    ext.undef_macros = ['_DEBUG']
        super().build_extensions()

use_cython = "--use-cython" in sys.argv
include_dirs = environ['include'].split(';')
suffix = "pyx" if use_cython else "c"
exts = [
    Extension(
        name='pyimg.core',
        sources=['src\\pyimg\\core.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pyimg.fft',
        sources=['src\\pyimg\\fft.' + suffix],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pyimg.utils',
        sources=['src\\pyimg\\utils.' + suffix],
        include_dirs=include_dirs,
    )
]
if use_cython:
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    cmdclass={'build_ext': build},
)
