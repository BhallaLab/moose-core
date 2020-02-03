__author__     = "Dilawar Singh"
__copyright__  = "Copyright 2019-, Dilawar Singh"
__maintainer__ = "Dilawar Singh"
__email__      = "dilawars@ncbs.res.in"

import os
import sys
import subprocess

#### TEST IF REQUIRED TOOLS EXISTS.
if sys.version_info[0] < 3:
    print("[ERROR] You must use python3.5 or higher. " 
        "You used %s" % sys.version + " which is not supported.")
    quit(-1)

try:
    cmakeVersion = subprocess.check_output(["cmake", "--version"])
    print("[INFO ] CMake found: %s" % cmakeVersion.decode('utf8'))
    
except Exception as e:
    print(f"[ERROR] cmake is not found. Please install cmake.", file=sys.stderr)
    quit(-1)


# See https://docs.python.org/3/library/distutils.html
# setuptools is preferred over distutils. And we are supporting python3 only.
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext 
import subprocess
import pathlib

# Global variables.

sdir_ = pathlib.Path().absolute()
builddir_ = sdir_ / '_build'
builddir_.mkdir(parents=True, exist_ok=True)
cmakeCacheFile_ = builddir_ / 'CMakeCache.txt'
if cmakeCacheFile_.exists():
    cmakeCacheFile_.unlink()


numCores_ = os.cpu_count() - 1

version_ = '3.1.2'

# importlib is available only for python3. Since we build wheels, prefer .so
# extension. This way a wheel built by any python3.x will work with any python3.

class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])

class build_ext_cmake(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        global numCores_
        global sdir_
        print("\n==========================================================\n")
        print("[INFO ] Building pymoose in %s ..." % builddir_)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DPYTHON_EXECUTABLE=%s' % sys.executable,
            '-DVERSION_MOOSE=%s' % version_,
            '-DCMAKE_BUILD_TYPE=%s' % config
        ]
        build_args = ['--config', config, '--', '-j%d' % numCores_]
        os.chdir(str(builddir_))
        self.spawn(['cmake', str(sdir_)] + cmake_args)
        if not self.dry_run: 
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(sdir_))

with open("README.md") as f:
    readme = f.read()


setup(
    name="pymoose",
    version=version_,
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
    # python2 specific version here as well.
    install_requires=['numpy'],
    package_dir={
        'moose': os.path.join('python', 'moose'),
        'rdesigneur': os.path.join('python', 'rdesigneur')
    },
    package_data={
        'moose': [
            '_moose.so', 'neuroml2/schema/NeuroMLCoreDimensions.xml',
            'chemUtil/rainbow2.pkl'
        ]
    },
    ext_modules=[CMakeExtension('.')],
    cmdclass={'build_ext': build_ext_cmake},
)
