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
set(client_osg_libs osg osgUtil osgDB osgGA osgViewer osgText OpenThreads)
set(client_boost_libs boost_filesystem boost_system boost_serialization boost_thread)
add_executable(glclient ${glclient_sources})
target_link_libraries(glclient ${client_osg_libs} ${client_boost_libs})
