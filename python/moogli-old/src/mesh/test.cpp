#include <osg/Vec3f>
#include <osg/Vec4f>
#include <osg/Geode>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include "mesh/CylinderMesh.hpp"

int
main(int argc, char ** argv)
{
  CylinderMesh  cylinder_mesh;
  osg::Geometry * geometry = cylinder_mesh( osg::Vec3f(0.0f, 0.0f, 0.0f)
                                          , 2.0f
                                          , 2.0f
                                          , 10.0f
                                          , osg::Vec3f(0.0f, 0.0f, 1.0f)
                                          , 20
                                          , osg::Vec4f(1.0f, 0.0f, 0.0f, 0.0f)
                                          );
  osg::Geode * geode = new osg::Geode();
  geode -> addDrawable(geometry);

  osgViewer::Viewer viewer;
 viewer.setCameraManipulator(new osgGA::TrackballManipulator());
  viewer.setSceneData(geode);
  return viewer.run();
}
