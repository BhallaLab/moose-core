#ifndef __PICK_HANDLER_HPP__
#define __PICK_HANDLER_HPP__

#include "includes.hpp"
#include "core/SelectInfo.hpp"

using namespace std;
using namespace osg;

class PickHandler : public osgGA::GUIEventHandler
{
public:
    // This virtual method must be overrode by subclasses.
    // virtual void doUserOperations(osgUtil::LineSegmentIntersector::Intersection& ) = 0;
    virtual bool handle( const osgGA::GUIEventAdapter& ea
                       , osgGA::GUIActionAdapter& aa
                       );
};

#endif /* __PICK_HANDLER_HPP__ */
