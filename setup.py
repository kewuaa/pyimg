from os import environ
from distutils.unixccompiler import UnixCCompiler

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


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


environ['cc'] = 'gcc'
environ['cxx'] = 'g++'
include_dirs = environ['include'].split(';')
exts = [
    Extension(
        name='pyimg.core',
        sources=['src\\pyimg\\core.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pyimg.fft',
        sources=['src\\pyimg\\fft.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        name='pyimg.utils',
        sources=['src\\pyimg\\utils.pyx'],
        include_dirs=include_dirs,
    )
]
setup(
    ext_modules=cythonize(exts, language_level=3),
    zip_safe=False,
    package_dir={'pyimg': 'src\\pyimg'},
    cmdclass={'build_ext': build},
)
