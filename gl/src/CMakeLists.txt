cmake_minimum_required(VERSION 2.6)
# set(data_structure_headers
#   AckPickData.h
#   GLcellResetData.h
#   GLcellProcData.h
#   ParticleData.h
#   SmoldynShapeData.h
#   GLviewResetData.h
#   GLshapeData.h
#   )

# set(compartment_data_headers
#   GLCompartmentCylinderData.h
#   GLCompartmentDiskData.h
#   GLCompartmentSphereData.h
#   GLCompartmentHemiData.h
#   GLCompartmentTriData.h
#   GLCompartmentRectData.h
#   )

set(glclient_sources
  GLclient.cpp
  TextBox.cpp
  GLCompartment.cpp
  GLCompartmentCylinder.cpp
  GLCompartmentSphere.cpp
  GLCompartmentHemi.cpp
  GLCompartmentDisk.cpp
  GLCompartmentTri.cpp
  GLCompartmentRect.cpp
  GLviewShape.cpp
  )

# set(glclient_headers
#   ${compartment_data_headers}
#   GLclient.h
#   TextBox.h
#   GLCompartment.h
#   GLCompartmentType.h
#   GLCompartmentCylinder.h
#   GLCompartmentSphere.h
#   GLCompartmentHemi.h
#   GLCompartmentDisk.h
#   GLCompartmentTri.h
#   GLCompartmentRect.h
#   ParticleData.h
#   SmoldynShapeData.h
#   GLviewShape.h
#   )
FIND_PACKAGE(Boost 1.40.0 COMPONENTS filesystem system serialization thread REQUIRED)
FIND_PACKAGE(OpenSceneGraph 2.8.0 COMPONENTS osgDB osgUtil osgGA osgViewer osgText)
INCLUDE_DIRECTORIES(${OPENSCENEGRAPH_INCLUDE_DIRS})
ADD_EXECUTABLE(glclient ${glclient_sources})
TARGET_LINK_LIBRARIES(glclient ${Boost_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES})

