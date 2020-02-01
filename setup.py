__author__     = "Dilawar Singh"
__copyright__  = "Copyright 2019-, Dilawar Singh"
__maintainer__ = "Dilawar Singh"
__email__      = "dilawars@ncbs.res.in"

import os
import sys
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

sdir_ = pathlib.Path().absolute()
numCores_ = os.cpu_count() - 1

version = '3.1.2'

# importlib is available only for python3. Since we build wheels, prefer .so
# extension. This way a wheel built by any python3.x will work with any python3.
suffix = '.so'
try:
    import importlib.machinery
    suffix = importlib.machinery.EXTENSION_SUFFIXES[-1]
except Exception as e:
    print('[WARN] Failed to determine importlib suffix')
    suffix = '.so'
print('[INFO] Suffix for python SO: %s' % suffix)

class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])

class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        global numCores_
        global sdir_

        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DPYTHON_EXECUTABLE=%s' % sys.executable,
            '-DCMAKE_BUILD_TYPE=%s' % config
        ]

        print("[INFO ] Building pymoose in %s ..." % build_temp)
        build_args = ['--config', config, '--', '-j%d' % numCores_]
        os.chdir(str(build_temp))
        self.spawn(['cmake', str(sdir_)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(sdir_))


with open("README.md") as f:
    readme = f.read()

setup(
    name="pymoose",
    version=version,
    description= 'Python scripting interface of MOOSE Simulator (https://moose.ncbs.res.in)',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='MOOSERes',
    author_email='bhalla@ncbs.res.in',
    maintainer='Dilawar Singh',
    maintainer_email='dilawars@ncbs.res.in',
    url='http://moose.ncbs.res.in',
    packages=[
        'rdesigneur', 'moose', 'moose.SBML', 'moose.genesis', 'moose.neuroml',
        'moose.neuroml2', 'moose.chemUtil', 'moose.chemMerge'
    ],
    install_requires=['numpy'],
    package_dir={
        'moose': os.path.join('python', 'moose'),
        'rdesigneur': os.path.join('python', 'rdesigneur')
    },
    package_data={
        'moose': [
            '_moose' + suffix, 'neuroml2/schema/NeuroMLCoreDimensions.xml',
            'chemUtil/rainbow2.pkl'
        ]
    },
    ext_modules=[CMakeExtension('.')],
    cmdclass={'build_ext': build_ext},
)
