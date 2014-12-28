# setup.py --- 
# 
# Filename: setup.py
# Description: 
# Author: subha
# Maintainer: 
# Created: Sun Dec  7 20:32:02 2014 (+0530)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.

# 
# 

# Code:
"""
This scripts compiles moose using python distutils module. As of Sat
Dec 27 14:41:33 EST 2014 , it works on cygwin 64 bit on Windows 7 with
the latest packages installed. Also tested was Ubuntu 13.04.

Pre-requisites:

You need to have Python-dev, numpy, libxml-dev, gsl-dev and hdf5-dev
libraries installed.

libSBML needs to be downloaded, built and installed separately.

"""
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX :: Linux",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering :: Bio-Informatics"]

NAME='moose'
DESCRIPTION = 'MOOSE is the Multiscale Object Oriented Simulation Environment'

fid = open('README', 'r')
long_description = fid.read()
fid.close()
idx = max(0, long_description.find('MOOSE is the'))
idx_end = long_description.find('# VERSION')
LONG_DESCRIPTION = long_description[idx:idx_end]

URL = 'http://moose.ncbs.res.in'
DOWNLOAD_URL = 'http://sourceforge.net/projects/moose/files/latest/download'
LICENSE = "GPLv3"
AUTHOR = '(alphabetically) Upinder Bhalla, Niraj Dudani, Aditya Gilra, Aviral Goel, GV Harsharani, Subhasis Ray and Dilawar Singh'
PLATFORMS = "Linux, Windows/cygwin"
VERSION = '3.0.1'

BUILD_TARGET = 'moose._moose'
SOURCES=['external/muparser/muParser.cpp',
         'external/muparser/muParserBase.cpp',
         'external/muparser/muParserTokenReader.cpp',
         'external/muparser/muParserError.cpp',
         'external/muparser/muParserCallback.cpp',
         'external/muparser/muParserBytecode.cpp',
         'basecode/consts.cpp',
         'basecode/Element.cpp',
         'basecode/DataElement.cpp',
         'basecode/GlobalDataElement.cpp',
         'basecode/LocalDataElement.cpp',
         'basecode/Eref.cpp',
         'basecode/Finfo.cpp',
         'basecode/DestFinfo.cpp',
         'basecode/Cinfo.cpp',
         'basecode/SrcFinfo.cpp',
         'basecode/ValueFinfo.cpp',
         'basecode/SharedFinfo.cpp',
         'basecode/FieldElementFinfo.cpp',
         'basecode/FieldElement.cpp',
         'basecode/Id.cpp',
         'basecode/ObjId.cpp',
         'basecode/global.cpp',
         'basecode/SetGet.cpp',
         'basecode/OpFuncBase.cpp',
         'basecode/EpFunc.cpp',
         'basecode/HopFunc.cpp', 
         'basecode/SparseMatrix.cpp',
         'basecode/doubleEq.cpp',
         'basecode/testAsync.cpp',
         'basecode/main.cpp',
         'biophysics/IntFire.cpp',
         'biophysics/SpikeGen.cpp',
         'biophysics/RandSpike.cpp',
         'biophysics/CompartmentDataHolder.cpp',
         'biophysics/CompartmentBase.cpp',
         'biophysics/Compartment.cpp',
         'biophysics/SymCompartment.cpp',
         'biophysics/GapJunction.cpp',
         'biophysics/ChanBase.cpp',
         'biophysics/ChanCommon.cpp',
         'biophysics/HHChannelBase.cpp',
         'biophysics/HHChannel.cpp',
         'biophysics/HHGate.cpp',
         'biophysics/HHGate2D.cpp',
         'biophysics/HHChannel2D.cpp',
         'biophysics/CaConcBase.cpp',
         'biophysics/CaConc.cpp',
         'biophysics/MgBlock.cpp',
         'biophysics/Nernst.cpp',
         'biophysics/Neuron.cpp',
         'biophysics/ReadCell.cpp',
         'biophysics/SynChan.cpp',
         'biophysics/NMDAChan.cpp',
         'biophysics/testBiophysics.cpp',
         'biophysics/IzhikevichNrn.cpp',
         'biophysics/DifShell.cpp',
         'biophysics/Leakage.cpp',
         'biophysics/VectorTable.cpp',
         'biophysics/MarkovRateTable.cpp',
         'biophysics/MarkovChannel.cpp',
         'biophysics/MarkovGslSolver.cpp',
         'biophysics/MatrixOps.cpp',
         'biophysics/MarkovSolverBase.cpp',
         'biophysics/MarkovSolver.cpp',
         'biophysics/VClamp.cpp',
         'builtins/Arith.cpp',
         'builtins/Group.cpp',
         'builtins/Mstring.cpp',
         'builtins/Func.cpp',
         'builtins/Function.cpp',
         'builtins/Variable.cpp',
         'builtins/TableBase.cpp',
         'builtins/Table.cpp',
         'builtins/Interpol.cpp',
         'builtins/StimulusTable.cpp',
         'builtins/TimeTable.cpp',
         'builtins/Stats.cpp',
         'builtins/SpikeStats.cpp',
         'builtins/Interpol2D.cpp',
         'builtins/HDF5WriterBase.cpp',
         'builtins/HDF5DataWriter.cpp',
         'builtins/testBuiltins.cpp',
         'device/PulseGen.cpp',
         'device/DiffAmp.cpp',
         'device/PIDController.cpp',
         'device/RC.cpp',
         'diffusion/FastMatrixElim.cpp',
         'diffusion/DiffPoolVec.cpp',
         'diffusion/Dsolve.cpp',
         'diffusion/testDiffusion.cpp',
         'hsolve/HSolveStruct.cpp',
         'hsolve/HinesMatrix.cpp',
         'hsolve/HSolvePassive.cpp',
         'hsolve/RateLookup.cpp',
         'hsolve/HSolveActive.cpp',
         'hsolve/HSolveActiveSetup.cpp',
         'hsolve/HSolveInterface.cpp',
         'hsolve/HSolve.cpp',
         'hsolve/HSolveUtils.cpp',
         'hsolve/testHSolve.cpp',
         'hsolve/ZombieCompartment.cpp',
         'hsolve/ZombieCaConc.cpp',
         'hsolve/ZombieHHChannel.cpp',
         'intfire/IntFireBase.cpp',
         'intfire/LIF.cpp',
         'intfire/QIF.cpp',
         'intfire/ExIF.cpp',
         'intfire/AdExIF.cpp',
         'intfire/AdThreshIF.cpp',
         'intfire/IzhIF.cpp',
         'intfire/testIntFire.cpp',
         'kinetics/PoolBase.cpp',
         'kinetics/Pool.cpp',
         'kinetics/BufPool.cpp',
         'kinetics/ReacBase.cpp',
         'kinetics/Reac.cpp',
         'kinetics/EnzBase.cpp',
         'kinetics/CplxEnzBase.cpp',
         'kinetics/Enz.cpp',
         'kinetics/MMenz.cpp',
         'kinetics/Species.cpp',
         'kinetics/ReadKkit.cpp',
         'kinetics/WriteKkit.cpp',
         'kinetics/ReadCspace.cpp',
         'kinetics/lookupVolumeFromMesh.cpp',
         'kinetics/testKinetics.cpp',
         'ksolve/KinSparseMatrix.cpp',
         'ksolve/ZombiePool.cpp',
         'ksolve/ZombiePoolInterface.cpp',
         'ksolve/ZombieBufPool.cpp',
         'ksolve/ZombieReac.cpp',
         'ksolve/ZombieEnz.cpp',
         'ksolve/ZombieMMenz.cpp',
         'ksolve/VoxelPoolsBase.cpp',
         'ksolve/VoxelPools.cpp',
         'ksolve/GssaVoxelPools.cpp',
         'ksolve/RateTerm.cpp',
         'ksolve/FuncTerm.cpp',
         'ksolve/Stoich.cpp',
         'ksolve/Ksolve.cpp',
         'ksolve/SteadyState.cpp',
         'ksolve/Gsolve.cpp',
         'ksolve/testKsolve.cpp',
         'mesh/ChemCompt.cpp',
         'mesh/MeshCompt.cpp',
         'mesh/MeshEntry.cpp',
         'mesh/CubeMesh.cpp',
         'mesh/CylBase.cpp',
         'mesh/CylMesh.cpp',
         'mesh/NeuroNode.cpp',
         'mesh/NeuroMesh.cpp',
         'mesh/SpineEntry.cpp',
         'mesh/SpineMesh.cpp',
         'mesh/PsdMesh.cpp',
         'mesh/testMesh.cpp',
         'mpi/PostMaster.cpp',
         'mpi/testMpi.cpp',
         'msg/Msg.cpp',
         'msg/DiagonalMsg.cpp',
         'msg/OneToAllMsg.cpp',
         'msg/OneToOneMsg.cpp',
         'msg/SingleMsg.cpp',
         'msg/SparseMsg.cpp',
         'msg/OneToOneDataIndexMsg.cpp',
         'msg/testMsg.cpp',
         'pymoose/moosemodule.cpp',
         'pymoose/mfield.cpp',
         'pymoose/vec.cpp',
         'pymoose/melement.cpp',
         'pymoose/test_moosemodule.cpp',
         'randnum/mt19937ar.cpp',
         'sbml/SbmlWriter.cpp',
         'sbml/SbmlReader.cpp',
         'scheduling/Clock.cpp',
         'scheduling/testScheduling.cpp',             
         'shell/Shell.cpp',
         'shell/ShellCopy.cpp',
         'shell/ShellThreads.cpp',
         'shell/LoadModels.cpp',
         'shell/SaveModels.cpp',
         'shell/Neutral.cpp',
         'shell/Wildcard.cpp',
         'shell/testShell.cpp',
         'signeur/Adaptor.cpp',
         'signeur/testSigNeur.cpp',
         'synapse/SynHandlerBase.cpp',
         'synapse/SimpleSynHandler.cpp',
         'synapse/STDPSynHandler.cpp',
         'synapse/Synapse.cpp',
         'synapse/STDPSynapse.cpp',
         'synapse/testSynapse.cpp',
         'utility/strutil.cpp',
         'utility/types.cpp',
         'utility/setupenv.cpp',
         'utility/numutil.cpp',
         'utility/Annotator.cpp',
         'utility/Vec.cpp',
         'benchmarks/benchmarks.cpp',
         'benchmarks/kineticMarks.cpp'
     ]

INCLUDE_DIRS=['/usr/include',
              '/usr/local/include',
              '/usr/lib/python2.7/site-packages/numpy/core/include', # needed explicitly on cygwin
              'external/muparser',
              'basecode',
              'biophysics',
              'builtins',
              'device',
              'diffusion',
              'hsolve',
              'intfire',
              'kinetics',
              'kk',
              'ksolve',
              'mesh',
              'mpi',
              'msg',
              'pymoose',
              'randnum',
              'sbml',
              'scheduling',
              'shell',
              'signeur',
              'synapse',
              'utility']

LIBRARIES = ['gsl',
             'gslcblas', # required to avoid undefined refs on import
             'hdf5',
             'sbml']

LIBRARY_DIRS = ['/usr/lib64', '/usr/lib', '/usr/local/lib']

DEFINE_MACROS = [('USE_GSL', None), 
                 ('USE_HDF5', None),
                 ('NDEBUG', None),
                 ('USE_NUMPY', None),
                 ('H5_NO_DEPRECATED_SYMBOLS', None),
                 ('PYMOOSE', None),
                 ('USE_SBML', None), 
                 ('USE_HDF5', None)]

# EXTRA_LINK_ARGS = ['-rdynamic']
PACKAGES = ['moose', 'moose.backend', 'moose.neuroml', 'mgui', 'mgui.plugins']
SCRIPTS = ['gui/mgui.py']
PACKAGE_DATA = {'moose': ['copyleft', 'README'], 'mgui': ['icons/*', 'colormaps/*', 'bioModels/*']}
REQUIRES = ['numpy', 'gsl', 'hdf5', 'libsbml'] # using full-dependency
# python-libsbml, although available on PyPI, does not build with pip
# install and gsl is a C library. The links are just for informing the
# users.
DEPENDENCY_LINKS = ['git+git://git.savannah.gnu.org/gsl.git',
                    'svn+svn://svn.code.sf.net/p/sbml/code/trunk']

EXTRAS_REQUIRE = ['matplotlib', 'PyQt4']

setup_info = dict(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    requires=REQUIRES,
    package_dir={'': 'python', 'mgui': 'gui'}
)
# try:
#     from setuptools import setup
#     from setuptools.extenion import Extension
#     setup_info['install_requires'] = INSTALL_REQUIRES
#     setup_info['extras_require'] = EXTRAS_REQUIRE
#     setup_info['dependency_links'] = DEPENDENCY_LINKS
# except ImportError:
#     from distutils.core import setup, Extension

from distutils.core import setup, Extension

moose_module = Extension(
    BUILD_TARGET,
    sources=SOURCES,
    include_dirs=INCLUDE_DIRS,
    libraries=LIBRARIES,
    library_dirs=LIBRARY_DIRS,
    runtime_library_dirs=LIBRARY_DIRS,
    define_macros=DEFINE_MACROS,
    # extra_link_args=EXTRA_LINK_ARGS
)
print moose_module.runtime_library_dirs

setup_info['ext_modules'] = [moose_module]
setup(**setup_info)



# 
# setup.py ends here
