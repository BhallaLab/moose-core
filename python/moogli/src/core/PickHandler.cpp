#include "core/PickHandler.hpp"

bool
PickHandler::handle( const osgGA::GUIEventAdapter& ea
                   , osgGA::GUIActionAdapter& aa
                   )
{
    if ( ea.getEventType()!=osgGA::GUIEventAdapter::RELEASE
       ||ea.getButton()!=osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON
       ||!(ea.getModKeyMask()&osgGA::GUIEventAdapter::MODKEY_CTRL)
       )
    {
        return false;
    }

    osgViewer::View* viewer = dynamic_cast<osgViewer::View*>(&aa);
    if ( viewer )
    {


        RECORD_INFO("Reaching Here!");

        osgUtil::LineSegmentIntersector::Intersections intersections;
        if (viewer -> computeIntersections(ea,intersections))
        {
            const osgUtil::LineSegmentIntersector::Intersection& hit =
                *intersections.begin();
            RECORD_INFO("Detected !");
            // return hit.drawable -> asGeometry();
        }

        // osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector =
        // new osgUtil::LineSegmentIntersector( osgUtil::Intersector::WINDOW
        //                                    , ea.getXnormalized()
        //                                    , ea.getYnormalized()
        //                                    );
        // osgUtil::IntersectionVisitor iv( intersector.get() );
        // viewer->getCamera()->accept( iv );
        // if ( intersector->containsIntersections() )
        // {
        //     osgUtil::LineSegmentIntersector::Intersection result =
        //         *(intersector->getIntersections().begin());
        //     RECORD_INFO("Found Intersection!");
        //     // doUserOperations( result );
        // }
    }
    return false;
}
