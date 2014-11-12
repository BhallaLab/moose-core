#ifndef __INCLUDES_HPP__
#define __INCLUDES_HPP__
#undef ANY
/******************************************************************************/
/* C++ STANDARD LIBRARY HEADERS                                               */
/******************************************************************************/
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <utility>
#include <sstream>

/******************************************************************************/
/* OSG HEADERS                                                                */
/******************************************************************************/
#include <osg/Shape>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Vec3d>
#include <osg/Vec4d>
#include <osg/ref_ptr>
#include <osgViewer/Viewer>
#include <osg/MatrixTransform>

#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/CompositeViewer>

#include <osgQt/GraphicsWindowQt>

#include <osgGA/TrackballManipulator>

#include <osg/ShadeModel>
#include <osg/Material>
#include <osg/LightSource>
#include <osg/Light>
#include <osg/StateSet>
#include <osg/Depth>
#include <osgFX/Outline>

/******************************************************************************/
/* QT HEADERS                                                                 */
/******************************************************************************/
#include <QTimer>
#include <QApplication>
#include <QGridLayout>
#include <QScrollArea>
#include <QMenu>
#include <QCursor>
#include <QDrag>

/******************************************************************************/
/* UTILITY HEADERS                                                            */
/******************************************************************************/
#include "constants.hpp"
#include "globals.hpp"
#include "utility/utilities.hpp"
#include "utility/record.hpp"
#include "utility/definitions.hpp"

#include "Python.h"
// #include "utility/conversions.hpp"
#define ANY void
#endif  /* __INCLUDES_HPP__ */
