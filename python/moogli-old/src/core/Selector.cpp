#include "core/Selector.hpp"

Selector::Selector( ref_ptr<MatrixTransform> matrix_transform /* QWidget * right_click_menu */
                  ) : menu(new QMenu()) //: _right_click_menu(right_click_menu)
                    , mode(-1)
{
    _matrix_transform = matrix_transform;
    osgFX::Outline * outline = new osgFX::Outline();
    outline -> setColor(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    outline -> setWidth(10.0);
    _matrix_transform -> addChild(outline);
    Geode   * selection   = new Geode();
    outline -> addChild(selection);
    _selection = selection;
     // this -> callback = callback;
}

bool
Selector::handle( const osgGA::GUIEventAdapter& ea
                , osgGA::GUIActionAdapter& aa
                )
{
    osgViewer::Viewer* viewer = dynamic_cast<osgViewer::Viewer*>(&aa);

    if ( !viewer )
    {
        return false;
    }

    // Geometry * geometry   = _get_intersection(ea,viewer);
    // cout << geometry;
    // bool blank_click      = geometry == nullptr;
    // if(!blank_click)
    // {
    //     RECORD_INFO("Compartment clicked " + geometry -> getName());
    // }
    // else
    // {
    //     // return true;
    //     // RECORD_INFO("Problem");
    // }

    if(select_info -> get_event_type() == 2)
    {
        // RECORD_INFO("Event Type => 2");
        return true;
    }

    // select_info -> set_event_type(0);
    // select_info -> set_id("");

    // if(ea.getEventType() == osgGA::GUIEventAdapter::RELEASE)
    // {
    //     RECORD_INFO(to_string(mode));
    //     if(mode == -1)
    //     {
    //         return false;
    //     }
    //     if(mode == 2)
    //     {
    //         menu -> exec(QCursor::pos());
    //     }
    //     if(mode == 3)
    //     {
    //         menu -> exec(QCursor::pos());
    //     }
    //     mode = -1;
    //     return true;
    // }

    // if(ea.getEventType() == osgGA::GUIEventAdapter::DRAG)
    // {
    //     QDrag *drag = new QDrag(this);
    //     // The data to be transferred by the drag and drop operation is contained in a QMimeData object
    //     QMimeData *data = new QMimeData;
    //     data->setText("This is a test");
    //     // Assign ownership of the QMimeData object to the QDrag object.
    //     drag->setMimeData(data);
    //     // Start the drag and drop operation
    //     drag->start();
    // }

    bool drag_event_occurred = ea.getEventType() & osgGA::GUIEventAdapter::DRAG;
    bool push_event_occurred = ea.getEventType() & osgGA::GUIEventAdapter::PUSH;
    bool release_event_occurred = ea.getEventType() & osgGA::GUIEventAdapter::RELEASE;
    bool left_mouse_button_pressed = ea.getButton() == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON;
    bool ctrl_key_pressed = ea.getModKeyMask() &  osgGA::GUIEventAdapter::MODKEY_CTRL;

    if(left_mouse_button_pressed && push_event_occurred && ctrl_key_pressed)
    {
        Geometry * geometry = _get_intersection(ea,viewer);
        bool blank_click    = geometry == nullptr;
        if(blank_click)
        {
            _deselect_everything();
            return false;
        }
        else
        {
            _select_compartment(geometry);
            RECORD_INFO("Compartment clicked " + geometry -> getName());
            // viewer -> emit_signal(geometry -> getName());
            select_info -> set_event_type(2);
            select_info -> set_id(geometry -> getName().c_str());
            return true;
        }
        return false;
    }

    if(release_event_occurred && left_mouse_button_pressed)
    {
        Geometry * geometry   = _get_intersection(ea,viewer);
        bool blank_click      = geometry == nullptr;
        bool selection_exists = false; //blank_click ? false : selections.find(geometry -> getName()) == selection.end()
        if(!ctrl_key_pressed)
        {
            if(blank_click)
            {
                _deselect_everything();
            }
            else
            {
                // RECORD_INFO("Select Compartment"); // do nothing
                _select_compartment(geometry);
                select_info -> set_event_type(1);
                select_info -> set_id(geometry -> getName().c_str());
                return true;
            }
        }
        return false;
        // else
        // {
        //     if(blank_click)
        //     {
        //         RECORD_INFO("Deselect Everything."); // do nothing
        //         // _deselect_everything();
        //     }
        //     else if(selection_exists)
        //     {
        //         RECORD_INFO("Select Neuron.");
        //         _select_neuron(geometry);
        //     }
        //     else
        //     {
        //         RECORD_INFO("Select Compartment"); // do nothing
        //         // _deselect_everything();
        //         _select_compartment(geometry);
        //     }
        // }
    }

    // if(ea.getEventType() == osgGA::GUIEventAdapter::PUSH)
    // {
    //     if(ea.getButton() == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
    //     {
    //         Geometry * geometry = _get_intersection(ea,viewer);
    //         if(geometry)
    //         {
    //             if
    //             {
    //                 _select_compartment(geometry);
    //                 mode = 0;
    //             }
    //             else
    //             {
    //                 _select_neuron(geometry);
    //                 mode = 1;
    //             }
    //             return true;
    //         }
    //         mode = -1;
    //         return false;
    //     }

    //     if(ea.getButton() == osgGA::GUIEventAdapter::RIGHT_MOUSE_BUTTON)
    //     {
    //         Geometry * geometry = _get_intersection(ea,viewer);
    //         if(geometry)
    //         {
    //             if(ea.getModKeyMask() &  osgGA::GUIEventAdapter::MODKEY_CTRL)
    //             {
    //                 _select_compartment(geometry);
    //                 mode = 2;
    //             }
    //             else
    //             {
    //                 _select_neuron(geometry);
    //                 mode = 3;
    //             }
    //             return true;
    //         }
    //         mode = -1;
    //         return false;
    //     }
    // }
    return false;
}

#if OPENSCENEGRAPH_MINOR_VERSION == 2

/*
http://comments.gmane.org/gmane.comp.graphics.openscenegraph.user/80993
*/
Geometry *
Selector::_get_intersection( const osgGA::GUIEventAdapter& ea
                           , osgViewer::Viewer* viewer
                           )
{
    osgUtil::LineSegmentIntersector::Intersections intersections;
    if (viewer -> computeIntersections(ea,intersections))
    {
        const osgUtil::LineSegmentIntersector::Intersection& hit =
            *intersections.begin();
        return hit.drawable -> asGeometry();
    }
    return nullptr;
}


/*
http://uncommoncode.wordpress.com/2010/09/22/select-objects-with-mouse-in-openscenegraph/
*/

#else

Geometry *
Selector::_get_intersection( const osgGA::GUIEventAdapter& ea
                           , osgViewer::Viewer * viewer
                           )
{
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector =
        new osgUtil::LineSegmentIntersector( osgUtil::Intersector::WINDOW
                                           , ea.getX()
                                           , ea.getY()
                                           );
    osgUtil::IntersectionVisitor iv( intersector.get() );
    viewer->getCamera()->accept( iv );

    if ( intersector->containsIntersections() )
    {
        const osgUtil::LineSegmentIntersector::Intersection& result =
                *(intersector->getIntersections().begin());

            // LOD * lod = dynamic_cast<LOD *>(result.drawable -> getParent(0) -> getParent(0));
        // RECORD_INFO("Reaching here!");
        return result.drawable -> asGeometry();
    }
    return nullptr;
}

#endif

void
Selector::_deselect()
{

}

void
Selector::_deselect_everything()
{
    _selection -> removeDrawables(0);
}


bool
Selector::_select_compartment(Geometry * geometry)
{
    _selection -> removeDrawables(0);
    _selection -> addDrawable(geometry);
    // Geode * geode = (Geode *)(geometry -> getParent(0));
    // Geode * geode = new Geode();
    // geode -> addDrawable();
    // LOD *   lod = (LOD *)(geode -> getParent(0)));
    // lod -> removeChild(geode);

    // osgFX::Outline * outline = new osgFX::Outline();
    // outline -> setColor(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    // outline -> addChild(geode);
    // lod -> add
    // matrix_transform -> addChild(outline);

    // if(geometry == nullptr)
    // {
    //     RECORD_INFO("Invalid click!");
    //     return false;
    // }
    // RECORD_INFO("Compartment clicked : " + geometry -> getName());
    return true;
}

bool
Selector::_select_neuron(Geometry * geometry)
{
    if(geometry == nullptr)
    {
        RECORD_INFO("Invalid click!");;
        return false;
    }
    RECORD_INFO("Compartment clicked => " + geometry -> getName());
    Geode * geode = (Geode *)(geometry -> getParent(0));
    RECORD_INFO("Neuron clicked => " + geode -> getName());
    ref_ptr<LOD>   lod((LOD *)(geode -> getParent(0)));
    MatrixTransform *matrix_transform = (MatrixTransform *)(lod -> getParent(0));
    matrix_transform -> removeChild(lod.get());
    osgFX::Outline * outline = new osgFX::Outline();
    outline -> setColor(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    outline -> addChild(lod.get());
    matrix_transform -> addChild(outline);
    return true;
}

// ref_ptr<LOD>
// Selector::_get_neuron()
// {

// }

// bool
// _select(LOD * lod)
// {

// }

// bool
// _select(Geometry * geometry)
// {
//     moose_id_t compartment_id = geometry -> getName();
//     moose_id_t neuron_id      =
// }

bool
_deselect(Geometry * geometry)
{
    return true;
}

