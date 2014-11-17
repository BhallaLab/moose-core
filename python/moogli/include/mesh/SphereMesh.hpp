#ifndef _SPHERE_MESH_HPP_
#define _SPHERE_MESH_HPP_

#include <osg/Shape>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Vec3d>
#include <osg/ShapeDrawable>
#include <osg/ref_ptr>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgViewer/ViewerEventHandlers>
#include <osgUtil/Optimizer>
#include <osgUtil/SmoothingVisitor>
#include <osg/LOD>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

using namespace std;
using namespace osg;

class SphereMesh
{
public:

    SphereMesh();

    Geometry *
    operator()( Vec3f        center = Vec3f(0.0f, 0.0f, 0.0f)
              , float        radius = 1.0f
              , unsigned int points = 10
              );

private:

    unordered_map< unsigned int
                 , const tuple< const Vec3Array * const
                              , const DrawElementsUShort * const
                              , const Vec3Array * const
                              >
                 > spheres;

    const tuple< const Vec3Array * const
               , const DrawElementsUShort * const
               , const Vec3Array * const
               >
    unit(unsigned int points = 10);
};

#endif /* _SPHERE_MESH_HPP_ */
