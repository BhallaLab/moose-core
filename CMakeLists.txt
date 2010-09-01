# Configuration to build MOOSE project 

#This CMakeLists.txt file has been made by putting together pieces of
# code from different sources: mostly Scribus.

cmake_minimum_required(VERSION 2.6)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE ON)
include(CPack) 
include(InstallRequiredSystemLibraries)
# We have some third-party CMake modules to find various packages.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules)

project(MOOSE)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MOOSE: the multiscale object-oriented simulation environment")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "3")
set(CPACK_PACKAGE_VERSION_PATCH "0")

set(VERSION_MAJOR 1)
set(VERSION_MINOR 3)
set(VERSION_PATCH 0)
set(VERSION_SUFFIX "svn")
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}.${VERSION_SUFFIX})
message(STATUS "${CMAKE_PROJECT_NAME} ${VERSION} will be built to install into ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_BINARY_DIRECTORY build)
#Our main directory is moose
set(MAIN_DIR_NAME "moose")

################################################################################
# Various options for the build
################################################################################
option(WANT_DEBUG
  "Build debug version"
  OFF)

option(WANT_PYMOOSE
  "Build PyMOOSE - MOOSE as Python module"
  ON)

option(USE_GSL
  "Use GNU Scientific Library"
  ON)

option(USE_GL
  "Enable OpenGL based 3-D visualization."
  ON)

option(USE_MPI
  "Build for parallel computers with Message Passing Interface(MPI) support."
  OFF)

option(USE_NEUROML
  "Enable support for models in neuroML format. (experimental)"
  ON)

option(GENERATE_WRAPPERS
  "Enable generation of Python wrapper classes (developers only)."
  OFF)
################################################################################
# Now we explicitly list the source files from each subdirectory.
# This is the recommended way, though apparently takes more lines of
# code as opposed to making cmake go through each subdirectory and
# FILE GLOB to include all .cpp files. This is to make sure :
#
# 0) Automatically run cmake when a new file is added to the build. If
# CMakeLists.txt is modified, make will call cmake first to update the
# make files.
#
# 1) the developers add only the files they are sure they want to
# include in the binary. Stray test files should not be automatically
# picked up.
#
# 2) the developers are responsible for the files they add.
#
# See this link for the logic behind this:
# http://www.cmake.org/cmake/help/cmake2.6docs.html#command:aux_source_directory
################################################################################
set(basecode_sources 
  basecode/Element.cpp
  basecode/Eref.cpp
  basecode/Id.cpp	
  basecode/IdGenerator.cpp	
  basecode/IdManager.cpp	
  basecode/Fid.cpp	
  basecode/RecvFunc.cpp 
  basecode/Ftype.cpp 
  basecode/Msg.cpp 
  basecode/SimpleElement.cpp	
  basecode/ArrayElement.cpp 
  basecode/Copy.cpp 
  basecode/Send.cpp 
  basecode/SrcFinfo.cpp 
  basecode/DestFinfo.cpp 
  basecode/ValueFinfo.cpp 
  basecode/LookupFinfo.cpp 
  basecode/ThisFinfo.cpp 
  basecode/DeletionMarkerFinfo.cpp 
  basecode/GlobalMarkerFinfo.cpp 
  basecode/Cinfo.cpp 
  basecode/DynamicFinfo.cpp 
  basecode/strconv.cpp 
  basecode/SharedFtype.cpp 
  basecode/SharedFinfo.cpp 
  basecode/SolveFinfo.cpp 
  basecode/setget.cpp 
  basecode/filecheck.cpp 
  basecode/ParallelDummy.cpp 
  basecode/Class.cpp	
  basecode/FuncVec.cpp 
  basecode/ExtFieldFinfo.cpp 
  basecode/UnitTests.cpp 
  basecode/TestBasecode.cpp   
  )

set(basecode_headers
  basecode/ArrayElement.h
  basecode/Cinfo.h
  basecode/Class.h
  basecode/DeletionMarkerFinfo.h
  basecode/DerivedFtype.h
  basecode/DestFinfo.h
  basecode/DynamicFinfo.h
  basecode/Element.h
  basecode/Eref.h
  basecode/ExtFieldFinfo.h
  basecode/Fid.h
  basecode/filecheck.h
  basecode/Finfo.h
  basecode/Ftype0.h
  basecode/Ftype1.h
  basecode/Ftype2.h
  basecode/Ftype3.h
  basecode/Ftype4.h
  basecode/Ftype5.h
  basecode/Ftype.h
  basecode/FuncVec.h
  basecode/GlobalMarkerFinfo.h
  basecode/header.h
  basecode/IdGenerator.h
  basecode/Id.h
  basecode/IdManager.h
  basecode/LookupFinfo.h
  basecode/LookupFtype.h
  basecode/moose.h
  basecode/Msg.h
  basecode/ProcInfo.h
  basecode/RecvFunc.h
  basecode/Send.h
  basecode/Serializer.h
  basecode/setget.h
  basecode/setgetLookup.h
  basecode/SharedFinfo.h
  basecode/SharedFtype.h
  basecode/SimpleElement.h
  basecode/Slot.h
  basecode/SolveFinfo.h
  basecode/SrcFinfo.h
  basecode/strconv.h
  basecode/ThisFinfo.h
  basecode/ValueFinfo.h
  basecode/ValueFtype.h
  )

set(connections_sources
  connections/ConnTainer.cpp	
  connections/SimpleConn.cpp	
  connections/One2AllConn.cpp	
  connections/All2OneConn.cpp	
  connections/Many2ManyConn.cpp	
  connections/One2OneMapConn.cpp	
  connections/TraverseMsgConn.cpp	
  connections/TraverseDestConn.cpp	
  )

set(connections_headers
  connections/Conn.h 
  connections/ConnTainer.h   
  connections/SimpleConn.h
  connections/One2AllConn.h
  connections/All2OneConn.h
  connections/One2OneMapConn.h
  connections/TraverseMsgConn.h
  connections/TraverseDestConn.h
  connections/Many2ManyConn.h
  )

set(maindir_sources
  maindir/main.cpp	
  maindir/nonblock.cpp 
  maindir/init.cpp 
  maindir/initCinfos.cpp 
  maindir/MuMPI.cpp
  )

set(maindir_headers
  maindir/init.h
  )

set(genesis_parser_sources
  genesis_parser/GenesisParser.cpp 
  genesis_parser/GenesisParserWrapper.cpp 
  genesis_parser/GenesisParser.tab.cpp 
  genesis_parser/symtab.cpp 
  genesis_parser/eval.cpp 
  genesis_parser/exec_fork.cpp 
  genesis_parser/parse.cpp 
  genesis_parser/getopt.cpp 
  genesis_parser/script.cpp 
  # genesis_parser/GenesisParser.yy.cpp
  )

set(genesis_parser_headers 
  genesis_parser/GenesisParser.h
  genesis_parser/GenesisParserWrapper.h
  genesis_parser/GenesisParser.tab.h
  )

if(WANT_MPI)
  set(genesis_parser_sources ${genesis_parser_sources} 	genesis_parser/ParGenesisParser.cpp)
endif(WANT_MPI)

set(shell_sources
  shell/Shell.cpp	
  shell/ParShell.cpp	
  shell/ReadCell.cpp 
  shell/SimDump.cpp 
  shell/LoadTab.cpp 
  shell/TestShell.cpp	
  )

set(shell_headers
  shell/Shell.h
  shell/ReadCell.h
  )

set(element_sources
  element/Neutral.cpp
  element/Wildcard.cpp  
  )

set(element_headers
  element/Neutral.h
  element/Wildcard.h  
  )

set(scheduling_sources 
  scheduling/ClockJob.cpp
  scheduling/SchedTests.cpp
  scheduling/Tick.cpp
  )

set(scheduling_headers
  scheduling/ClockJob.h
  scheduling/Tick.h
  )

if(WANT_MPI)
  set(scheduling_sources
    ${scheduling_sources}
    scheduling/ParTick.cpp)
  set(scheduling_headers
    ${scheduling_headers}
    scheduling/ParTick.h)
endif(WANT_MPI)
set(biophysics_sources
  biophysics/BinSynchan.cpp
  biophysics/BioScan.cpp
  biophysics/CaConc.cpp
  biophysics/Cell.cpp
  biophysics/Compartment.cpp
  biophysics/DifShell.cpp
  biophysics/GHK.cpp
  biophysics/HHChannel2D.cpp
  biophysics/HHChannel.cpp
  biophysics/HHGate2D.cpp
  biophysics/HHGate.cpp
  biophysics/IntFire.cpp
  biophysics/IzhikevichNrn.cpp
  biophysics/KinSynChan.cpp
  biophysics/Leakage.cpp
  biophysics/Mg_block.cpp
  # biophysics/MMPump.cpp
  biophysics/Nernst.cpp
  biophysics/NMDAChan.cpp
  biophysics/RandomSpike.cpp
  biophysics/script_out.cpp
  biophysics/SpikeGen.cpp
  biophysics/StochSynchan.cpp
  biophysics/SymCompartment.cpp
  # biophysics/SynChan3.cpp
  biophysics/SynChan.cpp
  biophysics/TauPump.cpp
  biophysics/TestBiophysics.cpp
  )
set(biophysics_headers
  biophysics/BinSynchan.h
  biophysics/BinSynInfo.h
  biophysics/BioScan.h
  biophysics/CaConc.h
  biophysics/Cell.h
  biophysics/Compartment.h
  biophysics/DifShell.h
  biophysics/GHK.h
  biophysics/HHChannel2D.h
  biophysics/HHChannel.h
  biophysics/HHGate2D.h
  biophysics/HHGate.h
  biophysics/IntFire.h
  biophysics/IzhikevichNrn.h
  biophysics/KinSynChan.h
  biophysics/Leakage.h
  biophysics/Mg_block.h
  # biophysics/MMPump.h
  biophysics/Nernst.h
  biophysics/NMDAChan.h
  biophysics/RandomSpike.h
  biophysics/script_out.h
  biophysics/SpikeGen.h
  biophysics/StochSynchan.h
  biophysics/StochSynInfo.h
  biophysics/SymCompartment.h
  # biophysics/SynChan3.h
  biophysics/SynChan.h
  biophysics/SynInfo.h
  biophysics/TauPump.h
  biophysics/VClamp.h
  )

set(hsolve_sources
  hsolve/HinesMatrix.cpp
  hsolve/HSolveActive.cpp
  hsolve/HSolveActiveSetup.cpp
  hsolve/HSolve.cpp
  hsolve/HSolveHub.cpp
  hsolve/HSolveInterface.cpp
  hsolve/HSolvePassive.cpp
  hsolve/HSolveStruct.cpp
  hsolve/RateLookup.cpp
  hsolve/TestHSolve.cpp
  )
set(hsolve_headers
  hsolve/HinesMatrix.h
  hsolve/HSolveActive.h
  hsolve/HSolve.h
  hsolve/HSolveHub.h
  hsolve/HSolvePassive.h
  hsolve/HSolveStruct.h
  hsolve/RateLookup.h
  hsolve/TestHSolve.h
  )

set(kinetics_sources
  kinetics/CylPanel.cpp
  kinetics/DiskPanel.cpp
  kinetics/Enzyme.cpp
  kinetics/Geometry.cpp
  kinetics/HemispherePanel.cpp
  kinetics/KinCompt.cpp
  kinetics/KinPlaceHolder.cpp
  kinetics/MathFunc.cpp
  kinetics/Molecule.cpp
  kinetics/Panel.cpp
  # kinetics/Particle.cpp
  kinetics/Reaction.cpp
  kinetics/RectPanel.cpp
  # kinetics/SmoldynHub.cpp
  kinetics/SpherePanel.cpp
  kinetics/Surface.cpp
  kinetics/TestKinetics.cpp
  kinetics/TriPanel.cpp
  )
set(kinetics_headers
  kinetics/CylPanel.h
  kinetics/DiskPanel.h
  kinetics/Enzyme.h
  kinetics/Geometry.h
  kinetics/HemispherePanel.h
  kinetics/InterHubFlux.h
  kinetics/KinCompt.h
  kinetics/KinPlaceHolder.h
  kinetics/MathFunc.h
  kinetics/Molecule.h
  kinetics/Panel.h
  # kinetics/Particle.h
  kinetics/Reaction.h
  kinetics/RectPanel.h
  # kinetics/SmoldynHub.h
  kinetics/SpherePanel.h
  kinetics/Surface.h
  kinetics/TriPanel.h
  )
set(ksolve_sources
  ksolve/GslIntegrator.cpp
  ksolve/GssaStoich.cpp
  ksolve/InterSolverFlux.cpp
  ksolve/KineticHub.cpp
  ksolve/KineticManager.cpp
  ksolve/KinSparseMatrix.cpp
  ksolve/Kintegrator.cpp
  ksolve/StateScanner.cpp
  ksolve/SteadyState.cpp
  ksolve/Stoich.cpp
  ksolve/TestKsolve.cpp
  )
set(ksolve_headers
  ksolve/GslIntegrator.h
  ksolve/GssaStoich.h
  ksolve/InterSolverFlux.h
  ksolve/KineticHub.h
  ksolve/KineticManager.h
  ksolve/KinSparseMatrix.h
  ksolve/Kintegrator.h
  ksolve/RateTerm.h
  ksolve/StateScanner.h
  ksolve/SteadyState.h
  ksolve/Stoich.h)

set(builtins_sources
  builtins/AscFile.cpp
  builtins/Calculator.cpp
  builtins/Interpol2D.cpp
  builtins/Interpol.cpp
  builtins/Table.cpp
  builtins/TimeTable.cpp)

set(builtins_headers
  builtins/AscFile.h
  builtins/Calculator.h
  builtins/Interpol2D.h
  builtins/Interpol.h
  builtins/Table.h
  builtins/TimeTable.h)

set(utility_headers
  utility/ArgParser.h
  utility/Configuration.h
  utility/NumUtil.h
  utility/PathUtility.h
  utility/Property.h
  utility/PropertyReader.h
  utility/SparseMatrix.h
  utility/StringUtil.h
  utility/Terminal.h
  utility/utility.h)

set(utility_sources
  utility/ArgParser.cpp
  utility/Configuration.cpp
  utility/help.cpp
  utility/Helpless.cpp
  utility/NumUtil.cpp
  utility/PathUtility.cpp
  utility/Property.cpp
  utility/PropertyReader.cpp
  utility/SparseMatrix.cpp
  utility/StringUtil.cpp)

set(randnum_sources
  randnum/Binomial.cpp
  randnum/BinomialRng.cpp
  randnum/Exponential.cpp
  randnum/ExponentialRng.cpp
  randnum/Gamma.cpp
  randnum/GammaRng.cpp
  randnum/mt19937ar.cpp
  randnum/Normal.cpp
  randnum/NormalRng.cpp
  randnum/Poisson.cpp
  randnum/PoissonRng.cpp
  randnum/Probability.cpp
  randnum/RandGenerator.cpp
  randnum/Uniform.cpp
  randnum/UniformRng.cpp)

set(randnum_headers
  randnum/Binomial.h
  randnum/BinomialRng.h
  randnum/Exponential.h
  randnum/ExponentialRng.h
  randnum/Gamma.h
  randnum/GammaRng.h
  randnum/Normal.h
  randnum/NormalRng.h
  randnum/Poisson.h
  randnum/PoissonRng.h
  randnum/Probability.h
  randnum/RandGenerator.h
  randnum/randnum.h
  randnum/Uniform.h
  randnum/UniformRng.h)

set(robots_sources
  robots/Adaptor.cpp
  robots/SigNeurAdaptors.cpp
  robots/SigNeur.cpp
  robots/SigNeurDiffusion.cpp
  robots/SigNeurNeuron.cpp
  robots/SigNeurSignaling.cpp)

set(robots_headers
  robots/Adaptor.h
  robots/SigNeur.h)

set(device_sources
  device/DiffAmp.cpp
  device/PIDController.cpp
  device/PulseGen.cpp
  device/RC.cpp)

set(device_headers
  device/DiffAmp.h
  device/PIDController.h
  device/PulseGen.h
  device/RC.h)

set(gl_sources
  gl/src/GLcell.cpp
  gl/src/GLshape.cpp
  gl/src/GLview.cpp)

set(gl_headers
  gl/src/GLCompartmentCylinderData.h
  gl/src/GLCompartmentDiskData.h
  gl/src/GLCompartmentSphereData.h 
  gl/src/GLCompartmentHemiData.h
  gl/src/GLCompartmentTriData.h
  gl/src/GLCompartmentRectData.h)

set(glclient_sources
  gl/src/GLclient.cpp
  gl/src/TextBox.cpp
  gl/src/GLCompartment.cpp
  gl/src/GLCompartmentCylinder.cpp
  gl/src/GLCompartmentSphere.cpp
  gl/src/GLCompartmentHemi.cpp
  gl/src/GLCompartmentDisk.cpp
  gl/src/GLCompartmentTri.cpp
  gl/src/GLCompartmentRect.cpp
  gl/src/GLviewShape.cpp)

set(glclient_headers
  gl/src/AckPickData.h
  gl/src/GLcellResetData.h
  gl/src/GLcellProcData.h
  gl/src/ParticleData.h
  gl/src/SmoldynShapeData.h
  gl/src/GLviewResetData.h
  gl/src/GLshapeData.h
  gl/src/GLclient.h
  gl/src/TextBox.h
  gl/src/GLCompartment.h
  gl/src/GLCompartmentType.h
  gl/src/GLCompartmentCylinder.h
  gl/src/GLCompartmentSphere.h
  gl/src/GLCompartmentHemi.h
  gl/src/GLCompartmentDisk.h
  gl/src/GLCompartmentTri.h
  gl/src/GLCompartmentRect.h
  gl/src/ParticleData.h
  gl/src/SmoldynShapeData.h
  gl/src/GLviewShape.h
  ${gl_headers})

set(pymoose_sources
  pymoose/PyMooseBase.cpp
  pymoose/PyMooseContext.cpp
  pymoose/Class.cpp
  pymoose/Cell.cpp
  pymoose/Compartment.cpp
  pymoose/Neutral.cpp
  pymoose/PyMooseUtil.cpp
  pymoose/HHChannel.cpp
  pymoose/HHGate.cpp
  pymoose/Mg_block.cpp
  pymoose/Interpol.cpp
  pymoose/CaConc.cpp
  pymoose/SpikeGen.cpp
  pymoose/PulseGen.cpp
  pymoose/RandomSpike.cpp
  pymoose/SynChan.cpp
  pymoose/NMDAChan.cpp
  pymoose/BinSynchan.cpp
  pymoose/StochSynchan.cpp
  pymoose/KinSynChan.cpp
  pymoose/Table.cpp
  pymoose/Nernst.cpp
  pymoose/Tick.cpp
  pymoose/ClockJob.cpp
  pymoose/Enzyme.cpp
  pymoose/KineticHub.cpp
  pymoose/Kintegrator.cpp
  pymoose/KineticManager.cpp
  pymoose/Molecule.cpp
  pymoose/Reaction.cpp
  pymoose/Stoich.cpp
  pymoose/NeuroScan.cpp
  pymoose/HSolve.cpp
  pymoose/RandGenerator.cpp
  pymoose/BinomialRng.cpp
  pymoose/ExponentialRng.cpp
  pymoose/GammaRng.cpp
  pymoose/NormalRng.cpp
  pymoose/PoissonRng.cpp
  pymoose/UniformRng.cpp
  pymoose/KinCompt.cpp
  pymoose/Panel.cpp
  pymoose/DiskPanel.cpp
  pymoose/HemispherePanel.cpp
  pymoose/RectPanel.cpp
  pymoose/CylPanel.cpp
  pymoose/SpherePanel.cpp
  pymoose/TriPanel.cpp
  pymoose/Surface.cpp
  pymoose/Geometry.cpp
  pymoose/MathFunc.cpp
  pymoose/AscFile.cpp
  pymoose/DifShell.cpp
  pymoose/GssaStoich.cpp
  pymoose/TauPump.cpp
  pymoose/GLcell.cpp
  pymoose/GLview.cpp
  pymoose/TimeTable.cpp
  pymoose/PIDController.cpp
  pymoose/RC.cpp
  pymoose/DiffAmp.cpp
  pymoose/GHK.cpp
  pymoose/IntFire.cpp
  pymoose/IzhikevichNrn.cpp
  pymoose/HHChannel2D.cpp
  pymoose/HHGate2D.cpp
  pymoose/SteadyState.cpp
  pymoose/InterSolverFlux.cpp
  pymoose/Adaptor.cpp
  pymoose/SigNeur.cpp
  pymoose/Leakage.cpp)

set(pymoose_headers
  pymoose/PyMooseUtil.h 
  pymoose/PyMooseContext.h 
  pymoose/PyMooseBase.h 
  pymoose/Compartment.h 
  pymoose/Neutral.h 
  pymoose/Class.h	
  pymoose/Cell.h	
  pymoose/Leakage.h	
  pymoose/HHChannel.h 
  pymoose/HHGate.h 
  pymoose/Mg_block.h	
  pymoose/Interpol.h 
  pymoose/CaConc.h 
  pymoose/SpikeGen.h 
  pymoose/RandomSpike.h	
  pymoose/PulseGen.h	
  pymoose/SynChan.h 
  pymoose/NMDAChan.h 
  pymoose/BinSynchan.h	
  pymoose/StochSynchan.h	
  pymoose/KinSynChan.h 
  pymoose/Table.h 
  pymoose/Nernst.h 
  pymoose/Tick.h 
  pymoose/ClockJob.h 
  pymoose/TableIterator.h 
  pymoose/Enzyme.h 
  pymoose/KineticHub.h 
  pymoose/Kintegrator.h 
  pymoose/KineticManager.h	
  pymoose/Molecule.h 
  pymoose/Reaction.h 
  pymoose/Stoich.h  
  pymoose/NeuroScan.h 
  pymoose/HSolve.h	
  pymoose/RandGenerator.h	
  pymoose/BinomialRng.h	
  pymoose/ExponentialRng.h	
  pymoose/GammaRng.h	
  pymoose/NormalRng.h	
  pymoose/PoissonRng.h	
  pymoose/UniformRng.h	
  pymoose/KinCompt.h	
  pymoose/Panel.h	
  pymoose/DiskPanel.h	
  pymoose/CylPanel.h	
  pymoose/HemispherePanel.h	
  pymoose/SpherePanel.h	
  pymoose/TriPanel.h	
  pymoose/RectPanel.h	
  pymoose/Surface.h	
  pymoose/Geometry.h	
  pymoose/MathFunc.h	
  pymoose/AscFile.h	
  pymoose/DifShell.h	
  pymoose/GssaStoich.h	
  pymoose/TauPump.h	
  pymoose/GLcell.h	
  pymoose/GLview.h	
  pymoose/TimeTable.h	
  pymoose/PIDController.h 
  pymoose/RC.h	
  pymoose/DiffAmp.h	
  pymoose/GHK.h	
  pymoose/IntFire.h	
  pymoose/IzhikevichNrn.h	
  pymoose/HHChannel2D.h	
  pymoose/HHGate2D.h	
  pymoose/SteadyState.h	
  pymoose/InterSolverFlux.h	
  pymoose/Adaptor.h	
  pymoose/SigNeur.h)

################################################################################
# Combine the source files into one global property.
################################################################################
get_property(is_defined GLOBAL PROPERTY MOOSE_SOURCES_LIST DEFINED)
if( NOT is_defined )
  define_property(GLOBAL PROPERTY MOOSE_SOURCES_LIST
    BRIEF_DOCS "List of source files"
    FULL_DOCS "List of all source(.c/.cpp) files to be compiled into MOOSE.")
endif(NOT is_defined)

# genesis_parser needs some preprocessor macros to be predefined.
set_property(SOURCE ${genesis_parser_sources} 
  PROPERTY 
  COMPILE_DEFINITIONS 
  YYMALLOC 
  YYFREE 
  YYSTYPE_IS_DECLARED)

    
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${basecode_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${connections_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${maindir_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${genesis_parser_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${element_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${shell_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${utility_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${randnum_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${scheduling_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${biophysics_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${hsolve_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${kinetics_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${ksolve_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${builtins_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${robots_sources}")
set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST "${device_sources}")

get_property(is_defined GLOBAL PROPERTY MOOSE_EXTERNAL_LIBRARIES DEFINED)
if( NOT is_defined )
  define_property(GLOBAL PROPERTY MOOSE_EXTERNAL_LIBRARIES
    BRIEF_DOCS "List of external libraries that are included conditionally."
    FULL_DOCS "List of external libraries for linking.")
endif(NOT is_defined)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lib)

################################################################################
# Check the build options to include additional libraries and build
# targets.
################################################################################
if(USE_GSL)
  find_package(GSL 1.10 REQUIRED)
  include_directories(${GSL_INCLUDE_DIR})
  set_property(GLOBAL PROPERTY APPEND PROPERTY MOOSE_EXTERNAL_LIBRARIES ${GSL_LIBRARIES})  
  set_property(SOURCE 
    ${ksolve_sources} 
    PROPERTY COMPILE_DEFINITIONS 
    USE_GSL)
  set_property(SOURCE
    ${randnum_sources}
    PROPERTY COMPILE_DEFINITIONS 
    USE_GSL)
endif(USE_GSL)


if(USE_GL)
  find_package(Boost 1.40.0 COMPONENTS filesystem system serialization thread REQUIRED)
  find_package(OpenSceneGraph 2.8.0 COMPONENTS osgDB osgUtil osgGA osgViewer osgText)
  include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
  set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST ${gl_sources})
  set_property(GLOBAL APPEND PROPERTY MOOSE_EXTERNAL_LIBRARIES ${OPENSCENEGRAPH_LIBRARIES} ${Boost_LIBRARIES})
  add_executable(glclient ${glclient_sources} ${glclient_headers})
  target_link_libraries(glclient ${Boost_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES})
endif(USE_GL)

if(USE_NEUROML)
  # add_subdirectory(external/neuroML_src)
  set_property(GLOBAL APPEND PROPERTY MOOSE_EXTERNAL_LIBRARIES neuroml)
endif(USE_NEUROML)

if(WANT_PYMOOSE)
  set_property(GLOBAL APPEND PROPERTY MOOSE_SOURCES_LIST ${pymoose_sources})
endif(WANT_PYMOOSE)

if(GENERATE_WRAPPERS)
  add_definitions(-DGENERATE_WRAPPERS)
endif(GENERATE_WRAPPERS)

# At the end of the day we need to add all subdirectories and the
# top-lvel directory to compiler's include path. Blame the
# inconsistent header include statements in MOOSE C++ code.
foreach( srcsubdir
    basecode
    connections
    maindir
    genesis_parser
    shell
    element
    scheduling
    biophysics
    hsolve
    kinetics
    ksolve
    builtins
    utility
    randnum
    robots
    device)
  include_directories(${srcsubdir})
endforeach(srcsubdir)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/include)



##############################################################################################################
########## check for the CPU we build for							    ##########

execute_process(
    COMMAND ${CMAKE_C_COMPILER} -dumpmachine
    OUTPUT_VARIABLE MACHINE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "(i?86-*)|(athlon-*)|(pentium-*)" _machine_x86 "${MACHINE}")
if(_machine_x86)
    message(STATUS "Found target X86")
    set(ARCH_X86 1)
endif(_machine_x86)

string(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _machine_x86_64 "${MACHINE}")
if(_machine_x86_64)
    message(STATUS "Found target X86_64")
    set(ARCH_X86_64 1)
endif(_machine_x86_64)

string(REGEX MATCH "(sparc64-*)|(SPARC64-*)" _machine_sparc_64 "${MACHINE}")
if(_machine_sparc_64)
    message(STATUS "Found target SPARC 64")
    set(ARCH_SPARC_64 1)
endif(_machine_sparc_64)

string(REGEX MATCH "(mips64-*)|(MIPS64-*)" _machine_mips_64 "${MACHINE}")
if(_machine_mips_64)
    message(STATUS "Found target MIPS 64")
    set(ARCH_MIPS_64 1)
endif(_machine_mips_64)

string(REGEX MATCH "(ppc-*)|(powerpc-*)" _machine_ppc "${MACHINE}")
if(_machine_ppc)
    message(STATUS "Found target PPC")
    set(ARCH_PPC 1)
endif(_machine_ppc)

string(REGEX MATCH "(ppc64-*)|(PPC64-*)|(powerpc64-*)" _machine_ppc_64 "${MACHINE}")
if(_machine_ppc_64)
    message(STATUS "Found target PPC64")
    set(ARCH_PPC_64 1)
endif(_machine_ppc_64)

string(REGEX MATCH "(sparc-*)" _machine_sparc "${MACHINE}")
if(_machine_sparc)
    message(STATUS "Found target Sparc")
    set(ARCH_SPARC 1)
endif(_machine_sparc)

string(REGEX MATCH "(sparcv9-*)" _machine_sparcv9 "${MACHINE}")
if(_machine_sparcv9)
    message(STATUS "Found target Sparc v9")
    set(ARCH_SPARCV9 1)
endif(_machine_sparcv9)

string(REGEX MATCH "(sparc64-*)" _machine_sparc64 "${MACHINE}")
if(_machine_sparc64)
    message(STATUS "Found target Sparc64")
    set(ARCH_SPARC64 1)
    set(ARCH64BIT 1)
endif(_machine_sparc64)

#convert any 64 bit build into generic 64 tag for below
if(ARCH_X86_64 OR ARCH_SPARC_64 OR ARCH_MIPS_64 OR ARCH_PPC_64) 
  set(ARCH64BIT 1)
endif(ARCH_X86_64 OR ARCH_SPARC_64 OR ARCH_MIPS_64 OR ARCH_PPC_64) 

message(STATUS "Building for target ${MACHINE}")

##############################################################################################################
########## Versioning Setup                                                                         ##########

#On Apple, we ignore the versioning tag so all items are "moose" not "moose-version"
if(NOT BUILD_OSX_BUNDLE)
  set(BUILD_OSX_BUNDLE ${APPLE} CACHE TYPE BOOL)
endif(NOT BUILD_OSX_BUNDLE)

#Simplify future conditionals for Apple
if(APPLE AND BUILD_OSX_BUNDLE)
  set(APPLEBUNDLE TRUE CACHE TYPE BOOL)
endif(APPLE AND BUILD_OSX_BUNDLE)

#Announce we cached a version request before, overridden below for OSX
if(TAG_VERSION)
  if(NOT APPLEBUNDLE)
    message(STATUS "Previously selected version tag: ${TAG_VERSION}")
  endif(NOT APPLEBUNDLE)
  set(WANT_VERSIONING 1)
  set(CACHED_VERSIONING 1)
endif(TAG_VERSION)

#Remove version tags on OSX so our bundle is Moose.app
if(APPLEBUNDLE)
  if(WANT_VERSIONING OR CACHED_VERSIONING)
    message(STATUS "Ignoring version tag on OSX when building bundle")
    set(WANT_VERSIONING 0)
    set(CACHED_VERSIONING 0)
  endif(WANT_VERSIONING OR CACHED_VERSIONING)
endif(APPLEBUNDLE)

#Store our version string if require
if(WANT_VERSIONING AND NOT CACHED_VERSIONING)
  if(NOT CUSTOM_VERSIONTAG)
    set(TAG_VERSION "-${VERSION}" CACHE TYPE STRING)
  else (NOT CUSTOM_VERSIONTAG)
    set(TAG_VERSION ${CUSTOM_VERSIONTAG} CACHE TYPE STRING)
  endif(NOT CUSTOM_VERSIONTAG)
endif(WANT_VERSIONING AND NOT CACHED_VERSIONING)

#############################################
########## Install/Directory Setup ##########

if(APPLEBUNDLE)
  if("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/Contents/?$")
    # all ok
  else ("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/Contents/?$")
    if("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/?$")
      # add "/Contents" subpath
      set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/Contents")
    else ("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/?$")
      # no bundle name spcified, use appname and version
      set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/moose${VERSION}.app/Contents")
    endif("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/?$")
  endif("${CMAKE_INSTALL_PREFIX}" MATCHES "[.]app/Contents/?$")

  message(STATUS "APPLEBUNDLE install prefix: ${CMAKE_INSTALL_PREFIX}")
endif(APPLEBUNDLE)


#Setup all the directories we will use
#MAN
cmake_policy(SET CMP0005 OLD)
set(MANDIR "share/man/")
set(SHAREDIR "share/${MAIN_DIR_NAME}${TAG_VERSION}/")
add_definitions("-DSHAREDIR=\\\"${CMAKE_INSTALL_PREFIX}/${SHAREDIR}\\\"")

#SHARE - use the default on Apple as TAG_VERSION is empty
if(TAG_VERSION OR BUILD_OSX_BUNDLE)
  set(DOCDIR "share/doc/${MAIN_DIR_NAME}${TAG_VERSION}/")
  set(SHAREDIR "share/${MAIN_DIR_NAME}${TAG_VERSION}/")
else(TAG_VERSION OR BUILD_OSX_BUNDLE)
  set(DOCDIR "share/doc/${MAIN_DIR_NAME}-${VERSION}/")
  set(SHAREDIR "share/${MAIN_DIR_NAME}-${TAG_VERSION}/")
endif(TAG_VERSION OR BUILD_OSX_BUNDLE)
set(DEMOSDIR "${SHAREDIR}/DEMOS")
set(GUIDIR "${SHAREDIR}/gui")
add_definitions("-DDOCDIR=\\\"${CMAKE_INSTALL_PREFIX}/${DOCDIR}\\\"")
add_definitions("-DEMOSDIR=\\\"${CMAKE_INSTALL_PREFIX}/${DEMOSDIR}\\\"")
add_definitions("-DGUIDIR=\\\"${CMAKE_INSTALL_PREFIX}/${GUIDIR}\\\"")

#ICON
if(WIN32)
  set(ICONDIR "share/icons/")
else(WIN32)
  set(ICONDIR "share/${MAIN_DIR_NAME}${TAG_VERSION}/icons/")
endif(WIN32)
add_definitions("-DICONDIR=\\\"${CMAKE_INSTALL_PREFIX}/${ICONDIR}\\\"")

#DESKTOP
set(DESKTOPDIR "share/applications/")
add_definitions("-DDESKTOPDIR=\\\"${CMAKE_INSTALL_PREFIX}/${DESKTOPDIR}\\\"")

# TODO: consider improving removing the following - Subha
# APPLICATION_DATA_DIR
# It allows specify the user configuration directory in the compile time.
# Example: I want to use special ~/.moose135svn config directory for my trunk
#          cmake -DAPPLICATION_DATA_DIR=.moose135svn ...
if(APPLICATION_DATA_DIR)
	message(STATUS "Using ApplicationDataDir ${APPLICATION_DATA_DIR} instead of the platform default.")
    add_definitions("-DAPPLICATION_DATA_DIR=\\\"${APPLICATION_DATA_DIR}\\\"")
else (APPLICATION_DATA_DIR)
    message(STATUS "Using standard ApplicationDataDir. You can change it with -DAPPLICATION_DATA_DIR")
endif(APPLICATION_DATA_DIR)

#LIB LOCATION - lib or lib64 normally
#LIB
if(WANT_LIB64)
  if(NOT ARCH_X86_64)
    message(STATUS "WARNING: BUILDING 64 Bit build on non 64 bit architecture")
  endif(NOT ARCH_X86_64)
  set(LIB_SUFFIX "64" CACHE TYPE STRING)
endif(WANT_LIB64)
set(LIB_DIR_NAME "lib${LIB_SUFFIX}")
set(LIBDIR "${LIB_DIR_NAME}/${MAIN_DIR_NAME}${TAG_VERSION}/")
add_definitions("-DLIBDIR=\\\"${CMAKE_INSTALL_PREFIX}/${LIBDIR}\\\"")

##############################################################################################################
########## Build Setup                                                                              ##########

#Convert our simpler command line option to the CMake style
#None, Debug, Release, .. or custom ones
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(WANT_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
endif(WANT_DEBUG)
if(WANT_RELEASEWITHDEBUG)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(WANT_RELEASEWITHDEBUG)
if(NOT WANT_DEBUG AND NOT WANT_RELEASEWITHDEBUG)
  set(CMAKE_BUILD_TYPE RELEASE)
endif(NOT WANT_DEBUG AND NOT WANT_RELEASEWITHDEBUG)



#Based on our build type, setup our build options
if(APPLE)
  # gcc/mac
  if(WANT_UNIVERSAL_BUNDLE)
    message("Setting Universal (PPC/i386) Architecture for OSX Build/Bundle")
    set(CMAKE_OSX_ARCHITECTURES "ppc;i386" )
    set(CMAKE_TRY_COMPILE_OSX_ARCHITECTURES "ppc;i386" )
    set(CMAKE_OSX_SYSROOT "/Developer/SDKs/MacOSX10.4u.sdk")
  else(WANT_UNIVERSAL_BUNDLE)
    if(ARCH_PPC OR ARCH_PPC_64)
      message("Setting PPC Architecture for OSX Build/Bundle")
      set(CMAKE_OSX_ARCHITECTURES "ppc" )
      set(CMAKE_TRY_COMPILE_OSX_ARCHITECTURES "ppc" )
    else(ARCH_PPC OR ARCH_PPC_64)
      message("Setting i386 Architecture for OSX Build/Bundle")
      set(CMAKE_OSX_ARCHITECTURES "i386" )
      set(CMAKE_TRY_COMPILE_OSX_ARCHITECTURES "i386" )   
    endif(ARCH_PPC OR ARCH_PPC_64)
  endif(WANT_UNIVERSAL_BUNDLE)
  message(STATUS "${CMAKE_OSX_ARCHITECTURES} ${CMAKE_OSX_SYSROOT}")
  set( ENV{MACOSX_DEPLOYMENT_TARGET} 10.4 ) 
  set( ENV{LD_PREBIND} 1 )
  set( ENV{LD_PREBIND_ALLOW_OVERLAP} 1 )
  if(WANT_DEBUG)
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
	set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
  endif(WANT_DEBUG)
  if(WANT_RELEASEWITHDEBUG)
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
  endif(WANT_RELEASEWITHDEBUG)
  if(NOT WANT_DEBUG AND NOT WANT_RELEASEWITHDEBUG)
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
	set(CMAKE_C_FLAGS_RELEASE "-O3 -fno-inline-functions -Wall -mmacosx-version-min=10.4")
  endif(NOT WANT_DEBUG AND NOT WANT_RELEASEWITHDEBUG)
else(APPLE)
  if(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")
    # Windows build with Visual Studio
    # Surely there's a better way to identify the compiler?
    set(CMAKE_CXX_FLAGS_DEBUG)
    set(CMAKE_C_FLAGS_DEBUG)
  else(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")
    # vanilla gcc
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -Wall -fstrict-aliasing")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -Wall -fstrict-aliasing")
    
    if(_machine_x86 OR _machine_x86_64)
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
      set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
    endif(_machine_x86 OR _machine_x86_64)
    
  endif(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")
endif(APPLE)

if(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")
  set(CMAKE_CXX_FLAGS_RELEASE)
  set(CMAKE_C_FLAGS_RELEASE)
else(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")
  if(NOT WANT_DEBUG)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -fPIC")
  endif(NOT WANT_DEBUG)
  if(${CMAKE_BUILD_TYPE} MATCHES ".*Release.*")
    add_definitions(-DNDEBUG)
   endif(${CMAKE_BUILD_TYPE} MATCHES ".*Release.*")
endif(${CMAKE_GENERATOR} MATCHES "^(Visual Studio|NMake).*")

if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN
                  -DAVOID_WIN32_FILEIO
                  -D_CRT_SECURE_NO_DEPRECATE
                  -DCOMPILE_PLUGIN_AS_DLL
  )
  set(BUILD_WINDOWS_VERSION 1)
  set(SC_USE_GDI 1)
  if(NOT SC_SHOULD_NOT_USE_GDIPLUS)
    set(SC_USE_GDIPLUS 1)
  endif(NOT SC_SHOULD_NOT_USE_GDIPLUS)
  set(DLL_USE_NATIVE_API 1)
endif(WIN32)

##############################################################################################################
########## Find Dependencies                                                                        ##########
if(WANT_PYMOOSE)
# add_subdirectory(pymoose)
find_package(PythonInterp REQUIRED)
if (PYTHON_FOUND)
  message(STATUS "Found Python: ${PYTHON_EXECUTABLE}")
endif(PYTHON_FOUND)
#<< Swig
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})


set(CMAKE_SWIG_FLAGS -modern -threads)

set_source_files_properties(pymoose/moose.i PROPERTIES CPLUSPLUS ON)
get_property(moose_sources GLOBAL PROPERTY MOOSE_SOURCES_LIST)
message(STATUS SRCS=${moose_sources})
get_property(moose_external_libs GLOBAL PROPERTY MOOSE_EXTERNAL_LIBRARIES)
swig_add_module(moose python pymoose/moose.i ${moose_sources})
swig_link_libraries(moose ${PYTHON_LIBRARIES} ${moose_external_libs})

#>> Swig

#<< PyQt

set (PYQT4_MIN_VERSION "4.3.0")
find_package(PyQt4 REQUIRED)
if(PYQT4_FOUND)
  message("PyQt4 Found OK")
else(PYQT4_FOUND) # This is not really fatal: but not yet sure of the policy about pymoose -- Subha
  message(FATAL_ERROR "No PyQt4 found. UBUNTU/DEBIAN USERS: If you have PyQt4 installed and still are getting this error, then please install python-qt4-dev package.")
endif(PYQT4_FOUND)
#>> PyQt
else(WANT_PYMOOSE)
  get_property(moose_sources GLOBAL PROPERTY MOOSE_SOURCES_LIST)
  message(STATUS SRCS=${moose_sources})
  get_property(moose_external_libs GLOBAL PROPERTY MOOSE_EXTERNAL_LIBRARIES)
  add_executable(moose ${moose_sources})
  target_link_libraries(moose ${moose_external_libs})
endif(WANT_PYMOOSE)

######################################################
# install(TARGTES moose ${INSTALL_TARGETS_DEFAULT_ARGS})
set(DEFAULT_PY_SUBDIR share/pyshared)
set(MOOSE_PY_SUBDIR ${DEFAULT_PY_SUBDIR})
set(MOOSE_BIN_SUBDIR ${DEFAULT_BIN_SUBDIR}     CACHE STRING "Subdirectory where executables will be installed")
