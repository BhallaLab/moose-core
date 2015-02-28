#include "core/KeyboardHandler.hpp"

bool
KeyboardHandler::handle( const osgGA::GUIEventAdapter& ea
                       , osgGA::GUIActionAdapter&      aa
                       )
{
    if( ea.getEventType() != osgGA::GUIEventAdapter::KEYDOWN ) { return false; }
    osgViewer::Viewer* viewer = dynamic_cast<osgViewer::Viewer*>(&aa);
    if ( !viewer )
    {
        return false;
    }

    auto key = ea.getKey();
    switch(key)
    {
        case osgGA::GUIEventAdapter::KEY_Up     :   translate_up(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Down   :   translate_down(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Left   :   translate_left(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Right  :   translate_right(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Plus   :   translate_forward(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Minus  :   translate_backward(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Greater:   translate_forward(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Less   :   translate_backward(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_X      :   clockwise_roll(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Y      :   clockwise_pitch(*viewer);
                                                    return true;
        case osgGA::GUIEventAdapter::KEY_Z      :   clockwise_yaw(*viewer);
                                                    return true;

    }
    return false;
}

void
KeyboardHandler::translate_up(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Moving Up!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    view_matrix.getLookAt(eye,centre,up);
    osg::Vec3d translation = view_matrix.getTrans();
    translation.y() = translation.y() + translate_up_factor;
    view_matrix.setTrans(translation);
    viewer.getCamera()->setViewMatrix(view_matrix);
    // view_matrix.getLookAt(eye,centre,up);
    // viewer.getCameraManipulator() -> setByMatrix(view_matrix);
    // ((osgGA::OrbitManipulator *)(viewer.getCameraManipulator()))
    //     -> setTransformation(eye, centre, up);
    // viewer.getCameraManipulator() -> updateCamera(* viewer.getCamera());

    // viewer.frame();
}


void
KeyboardHandler::translate_down(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Moving Down!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    view_matrix.getLookAt(eye,centre,up);
    osg::Vec3d translation = view_matrix.getTrans();
    translation.y() = translation.y() - translate_down_factor;
    view_matrix.setTrans(translation);
    viewer.getCamera() -> setViewMatrix(view_matrix);
    // view_matrix.getLookAt(eye,centre,up);
    // viewer.getCameraManipulator() -> setByMatrix(view_matrix);
    // ((osgGA::OrbitManipulator *)())
    //     -> setTransformation(eye, centre, up);

    // viewer.getCameraManipulator() -> setTransformation(eye, center, up);
}

void
KeyboardHandler::translate_left(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Move Left!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    osg::Vec3d translation = view_matrix.getTrans();
    translation.x() = translation.x() - translate_left_factor;
    view_matrix.setTrans(translation);
    viewer.getCamera()->setViewMatrix(view_matrix);
    // view_matrix.getLookAt(eye,centre,up);
    // viewer.getCameraManipulator() -> setTransformation(eye, center, up);
}

void
KeyboardHandler::translate_right(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Move Right!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    osg::Vec3d translation = view_matrix.getTrans();
    translation.x() = translation.x() + translate_right_factor;
    view_matrix.setTrans(translation);
    viewer.getCamera()->setViewMatrix(view_matrix);
    // view_matrix.getLookAt(eye,centre,up);
    // viewer.getCameraManipulator() -> setTransformation(eye, center, up);
}

void
KeyboardHandler::translate_forward(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming in!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    view_matrix.getLookAt(eye,centre,up);
    osg::Vec3f look_vector = centre - eye;
    //look_vector.normalize();
    look_vector.normalize();
    eye = eye + look_vector / translate_forward_factor;
    viewer.getCamera()->setViewMatrixAsLookAt(eye, centre, up);
    // viewer.getCameraManipulator() -> setByMatrix(eye, center, up);
}

void
KeyboardHandler::translate_backward(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming out!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;

    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    view_matrix.getLookAt(eye,centre,up);
    osg::Vec3f look_vector = centre - eye;
    //look_vector.normalize();
    look_vector.normalize();
    eye = eye - look_vector * translate_backward_factor;
    viewer.getCamera()->setViewMatrixAsLookAt(eye, centre, up);
    // viewer.getCameraManipulator() -> setTransformation(eye, center, up);

}

void
KeyboardHandler::clockwise_roll(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming out!");

    osg::Vec3f eye;
    osg::Vec3f centre;
    osg::Vec3f up;
    osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    view_matrix.getLookAt(eye,centre,up);
    // Quat rotation;
    // Vec3d translation;
    // Vec3d scale;
    // view_matrix.decompose()
    Quat rotation = view_matrix.getRotate();
    Vec3d scale = view_matrix.getScale();
    rotation.makeRotate(clockwise_roll_factor,up);
    osg::Matrix new_view_matrix = osg::Matrix::scale(scale);
    new_view_matrix.setRotate(rotation);
    new_view_matrix.setTrans(centre);
    viewer.getCamera()->setViewMatrix( new_view_matrix);

    // osg::Vec3f look_vector = centre - eye;
    // // look_vector.normalize();
    // auto length = look_vector.length();
    // new_look_vector.normalize();
    // osg::Quat quaternion(clockwise_roll_factor,up);
    // osg::Vec3f new_look_vector = quaternion * look_vector;
    // new_look_vector.normalize();
    // new_look_vector *= length;
    // viewer.getCamera()->setViewMatrixAsLookAt(centre - new_look_vector, centre, up);

    //
    // osg::Vec3d translation = view_matrix.getTrans();
    // view_matrix.setRotate(quaternion);
    // view_matrix.setTrans(trans);
    // viewer.getCamera()->setViewMatrix(view_matrix);
}

void
KeyboardHandler::counterclockwise_roll(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming out!");
    // osg::Matrix view_matrix =viewer.getCamera()->getViewMatrix();
    // osg::Vec3d translation = view_matrix.getTrans();
    // view_matrix.getLookAt(eye,centre,up);
    // osg::Quat quaternion(-counterclockwise_roll_factor,up);
    // view_matrix.setRotate(quaternion);
    // view_matrix.setTrans(trans);
    // viewer.getCamera()->setViewMatrix(view_matrix);
}


void
KeyboardHandler::clockwise_pitch(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming out!");
}

void
KeyboardHandler::clockwise_yaw(osgViewer::Viewer & viewer)
{
    RECORD_INFO("Zooming out!");

}
