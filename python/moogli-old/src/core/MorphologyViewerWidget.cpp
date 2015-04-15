#include "core/MorphologyViewerWidget.hpp"

MorphologyViewerWidget::MorphologyViewerWidget( Morphology * morphology
                                            , QWidget * parent
                                            , const QGLWidget* share_widget
                                            , Qt::WindowFlags f
                                            ) : morphology(morphology)
                                              , QGLWidget( parent
                                                         , share_widget
                                                         , f
                                                         )
                                              , _graphics_window( new osgViewer::GraphicsWindowEmbedded( this->x()
                                                                                                       , this->y()
                                                                                                       , this->width()
                                                                                                       , this->height()
                                                                                                       )
                                                                )
                                              , _viewer( new osgViewer::CompositeViewer )
                                              , capture_format("jpeg")
                                              , capture_location("/home/aviral/moogli_test")
{

    osg::StateSet* stateSet = morphology -> get_scene_graph()->getOrCreateStateSet();
    osg::Material* material = new osg::Material;

    material->setColorMode( osg::Material::AMBIENT_AND_DIFFUSE );

    stateSet->setAttributeAndModes( material, osg::StateAttribute::ON );
    stateSet->setMode( GL_DEPTH_TEST, osg::StateAttribute::ON );

    _previous_width  = this -> width();
    _previous_height = this -> height();
    add_view(0,0,this -> width(), this -> height());
    // add_view(this -> width()/2, 0, this -> width()/2, this -> height());
    _viewer->setThreadingModel( osgViewer::CompositeViewer::SingleThreaded );

    // This ensures that the widget will receive keyboard events. This focus
    // policy is not set by default. The default, Qt::NoFocus, will result in
    // keyboard events that are ignored.
    this->setFocusPolicy( Qt::StrongFocus );
    this->setMinimumSize( 200
                        , 200
                        );

    // Ensures that the widget receives mouse move events even though no
    // mouse button has been pressed. We require this in order to let the
    // graphics window switch viewports properly.
    this->setMouseTracking( true );
    const osg::BoundingSphere& bs = morphology -> get_scene_graph() -> getBound();
    up_distance         = bs.radius() * 0.01;
    down_distance       = bs.radius() * 0.01;
    left_distance       = bs.radius() * 0.01;
    right_distance      = bs.radius() * 0.01;
    forward_distance    = bs.radius() * 0.01;
    backward_distance   = bs.radius() * 0.01;
    zoom_factor         = 0.10;
    roll_angle          = M_PI / 36.0;
    pitch_angle         = M_PI / 36.0;
    yaw_angle           = M_PI / 36.0;
}

Morphology *
MorphologyViewerWidget::get_morphology()
{
    return morphology;
}

void
MorphologyViewerWidget::split_horizontally( unsigned int view_index
                                          , unsigned int width_factor
                                          )
{
    float aspect_ratio = static_cast<float>( this->width()) / static_cast<float>( this->height() );
    osgViewer::View * parent_view = _viewer -> getView (view_index);
    osg::Camera * parent_camera = parent_view -> getCamera();
    osg::Viewport * parent_viewport = parent_camera -> getViewport();
    int new_parent_width = parent_viewport -> width() * (width_factor - 1) / width_factor;
    add_view( parent_viewport -> x() + new_parent_width
            , parent_viewport -> y()
            , parent_viewport -> width() - new_parent_width
            , parent_viewport -> height()
            );
    parent_camera->setViewport( parent_viewport -> x()
                              , parent_viewport -> y()
                              , new_parent_width
                              , parent_viewport -> height()
                              );
    parent_camera->setProjectionMatrixAsPerspective( 30.f
                                                   , static_cast<float>(new_parent_width)
                                                   / static_cast<float>(parent_viewport -> height())
                                                   , 1.0f
                                                   , 10000.0f
                                                   );
}

void
MorphologyViewerWidget::split_vertically( unsigned int view_index
                                        , unsigned int height_factor
                                        )
{
    float aspect_ratio = static_cast<float>( this->width()) / static_cast<float>( this->height() );
    osgViewer::View * parent_view = _viewer -> getView (view_index);
    osg::Camera * parent_camera = parent_view -> getCamera();
    osg::Viewport * parent_viewport = parent_camera -> getViewport();

    // std::cout << "parent x      => "    << parent_viewport -> x() << std::endl;
    // std::cout << "parent y      => "    << parent_viewport -> y() << std::endl;
    // std::cout << "parent width  => "    << parent_viewport -> width() << std::endl;
    // std::cout << "parent height => "    << parent_viewport -> height() << std::endl;

    int new_parent_height = parent_viewport -> height() * (height_factor - 1) / height_factor;
    add_view( parent_viewport -> x()
            , parent_viewport -> y() + new_parent_height
            , parent_viewport -> width()
            , parent_viewport -> height() - new_parent_height
            );

    // std::cout << "child x      => "    << parent_viewport -> x() << std::endl;
    // std::cout << "child y      => "    << parent_viewport -> y() + new_parent_height << std::endl;
    // std::cout << "child width  => "    << parent_viewport -> width() << std::endl;
    // std::cout << "child height => "    << parent_viewport -> height() - new_parent_height << std::endl;

    parent_camera->setViewport( parent_viewport -> x()
                              , parent_viewport -> y()
                              , parent_viewport -> width()
                              , new_parent_height
                              );
    // std::cout << "parent2 x      => "    << parent_viewport -> x() << std::endl;
    // std::cout << "parent2 y      => "    << parent_viewport -> y() << std::endl;
    // std::cout << "parent2 width  => "    << parent_viewport -> width() << std::endl;
    // std::cout << "parent2 height => "    << new_parent_height << std::endl;

    parent_camera->setProjectionMatrixAsPerspective( 30.f
                                                   , static_cast<float>(parent_viewport -> width())
                                                   / static_cast<float>(new_parent_height)
                                                   , 1.0f
                                                   , 10000.0f
                                                   );
}

void
MorphologyViewerWidget::add_view( int x
                                , int y
                                , int width
                                , int height
                                )
{
    float aspect_ratio = static_cast<float>(width)
                      / static_cast<float>(height);

    osg::Camera* camera = new osg::Camera;
    camera->setViewport(x, y, width, height );
    camera->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.f ));
    camera->setProjectionMatrixAsPerspective( 30.f
                                            , aspect_ratio
                                            , 1.0f
                                            , 10000.0f
                                            );
    camera->setGraphicsContext( _graphics_window );
    osg::StateSet* stateset = camera -> getOrCreateStateSet();
    stateset->setGlobalDefaults();
    osgViewer::View* view = new osgViewer::View();
    view->setCamera( camera );
    view->setSceneData( morphology -> get_scene_graph() );
    // view->addEventHandler( new osgViewer::StatsHandler );
    view->setCameraManipulator( new osgGA::TrackballManipulator() );
    _viewer->addView( view );
    unsigned int index = _viewer -> getNumViews() - 1;
    QDir().mkdir((capture_location + "/" + to_string(index)).c_str());
    auto* capture_operation =
        new osgViewer::ScreenCaptureHandler::WriteToFile( capture_location + "/" + to_string(index)
                                                        , capture_format
                                                        );

    // auto* capture_handler = new MorphologyCaptureHandler(
    //     dynamic_cast<osgViewer::ScreenCaptureHandler::CaptureOperation *>(capture_operation)
    //                                                            );
    // view -> addEventHandler(capture_handler);
}

MorphologyViewerWidget::~MorphologyViewerWidget()
{
}

void
MorphologyViewerWidget::paintEvent( QPaintEvent* /* paintEvent */ )
{
    this->makeCurrent();
    QPainter painter( this );
    painter.setRenderHint( QPainter::Antialiasing );
    this->paintGL();
    painter.end();
    this->swapBuffers();
    this->doneCurrent();
}

void
MorphologyViewerWidget::paintGL()
{
    _viewer -> frame();
}

void
MorphologyViewerWidget::resizeGL( int width, int height )
{
    this->getEventQueue() ->  windowResize( this->x(), this->y(), width, height );
    _graphics_window      ->  resized( this->x(), this->y(), width, height );
    this->onResize( width, height );
}

unsigned int
MorphologyViewerWidget::_get_view_index_with_focus()
{
    osgViewer::ViewerBase::Views views;
    osgViewer::View * view = _viewer -> getViewWithFocus();
    _viewer -> getViews(views);
    auto iter = std::find(views.begin(),views.end(),view);
    if(iter == views.end()) { return 0; }
    return (iter - views.begin());
}

void
MorphologyViewerWidget::keyPressEvent( QKeyEvent* event )
{
    QString keyString   = event->text();
    const char* keyData = keyString.toLocal8Bit().data();
    unsigned int view_index = _get_view_index_with_focus();
    // std::cout << "Index => " << view_index << std::endl;
    switch(event->key())
    {
        case Qt::Key_Q          :   if(event->modifiers() & Qt::ControlModifier)
                                    {
                                        QApplication::quit();
                                    }
        // case Qt::Key_P          :   if(event->modifiers() & Qt::ControlModifier)
        //                             {
        //                                 QApplication::quit();
        //                             }
        // case Qt::Key_S          :   if(event->modifiers() & Qt::ShiftModifier)
        //                             {
        //                                 toggle_horizontal_split();
        //                             }
        // case Qt::Key_Z          :   if(event->modifiers() & Qt::ShiftModifier)
        //                             {
        //                                 toggle_vertical_split();
        //                             }
        // case Qt::Key_S          :   if(event->modifiers() & Qt::ShiftModifier)
        //                             {
        //                                 capture_continuous_toggle(view_index);
        //                             }
        //                             else
        //                             {
        //                                 capture_once(view_index);
        //                             }
                                    break;
        case Qt::Key_Space      :   home(view_index);
                                    break;
        case Qt::Key_Up         :   up(up_distance, view_index);
                                    break;
        case Qt::Key_Down       :   down(down_distance, view_index);
                                    break;
        case Qt::Key_Left       :   left(left_distance, view_index);
                                    break;
        case Qt::Key_Right      :   right(right_distance, view_index);
                                    break;
        case Qt::Key_F          :   forward(forward_distance, view_index);
                                    break;
        case Qt::Key_B          :   backward(backward_distance, view_index);
                                    break;
        case Qt::Key_Plus       :   zoom(zoom_factor, view_index);
                                    break;
        case Qt::Key_Equal      :   zoom(zoom_factor, view_index);
                                    break;
        case Qt::Key_Period     :   zoom(zoom_factor, view_index);
                                    break;
        case Qt::Key_Greater    :   zoom(zoom_factor, view_index);
                                    break;
        case Qt::Key_Minus      :   zoom(-zoom_factor, view_index);
                                    break;
        case Qt::Key_Underscore :   zoom(-zoom_factor, view_index);
                                    break;
        case Qt::Key_Comma      :   zoom(-zoom_factor, view_index);
                                    break;
        case Qt::Key_Less       :   zoom(-zoom_factor, view_index);
                                    break;
        case Qt::Key_R          :   if(event->modifiers() & Qt::ShiftModifier)
                                    {
                                        roll(-roll_angle, view_index);
                                    }
                                    else
                                    {
                                        roll(roll_angle, view_index);
                                    }
                                    break;
        case Qt::Key_P          :   if(event->modifiers() & Qt::ShiftModifier)
                                    {
                                        pitch(-pitch_angle, view_index);
                                    }
                                    else
                                    {
                                        pitch(pitch_angle, view_index);
                                    }
                                    break;
        case Qt::Key_Y          :   if(event->modifiers() & Qt::ShiftModifier)
                                    {
                                        yaw(-yaw_angle, view_index);
                                    }
                                    else
                                    {
                                        yaw(yaw_angle, view_index);
                                    }
                                    break;
        default                 :   break;
    }
}

// void
// MorphologyViewerWidget::capture_continuous_toggle(unsigned int index)
// {
//     osgViewer::View * view = _viewer -> getView(index);
//     auto * handler = dynamic_cast<MorphologyCaptureHandler *>((view -> getEventHandlers()).front().get());
//     handler -> toggle_capture();
// }

// void
// MorphologyViewerWidget::capture_once(unsigned int index)
// {
//     osgViewer::View * view = _viewer -> getView(index);
//     auto * handler = dynamic_cast<MorphologyCaptureHandler *>((view -> getEventHandlers()).front().get());
//     handler->setFramesToCapture(1);
// }

void
MorphologyViewerWidget::home(unsigned int index)
{
    osgViewer::View * view = _viewer -> getView(index);
    osgGA::TrackballManipulator * manipulator = dynamic_cast<osgGA::TrackballManipulator *>(view -> getCameraManipulator());
    manipulator -> home(0.0);
}

void
MorphologyViewerWidget::forward(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye + look * distance_delta, center + look * distance_delta, up);
}

void
MorphologyViewerWidget::backward(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye - look * distance_delta, center - look * distance_delta, up);
}

void
MorphologyViewerWidget::left(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye + side * distance_delta, center + side * distance_delta, up);
}

void
MorphologyViewerWidget::right(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye - side * distance_delta, center - side * distance_delta, up);
}

void
MorphologyViewerWidget::up(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye - up * distance_delta, center - up * distance_delta, up);
}

void
MorphologyViewerWidget::down(double distance_delta, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye + up * distance_delta, center + up * distance_delta, up);
}

void
MorphologyViewerWidget::zoom(double factor, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    _set_transformation(index, eye + look * factor * distance, center, up);
}

void
MorphologyViewerWidget::_get_transformation( unsigned int index
                                           , osg::Vec3d  & eye
                                           , osg::Vec3d  & center
                                           , double      & distance
                                           , osg::Vec3d  & up
                                           , osg::Vec3d  & look
                                           , osg::Vec3d  & side
                                           )
{
    osgViewer::View * view = _viewer -> getView(index);
    osgGA::TrackballManipulator * manipulator = dynamic_cast<osgGA::TrackballManipulator *>(view -> getCameraManipulator());
    manipulator -> getTransformation(eye, center, up);
    up.normalize();
    look = center - eye;
    distance = look.normalize();
    side = look ^ up;
    side.normalize();
}

void
MorphologyViewerWidget::_set_transformation( unsigned int index
                                           , const osg::Vec3d & eye
                                           , const osg::Vec3d & center
                                           , const osg::Vec3d & up
                                           )
{
    osgViewer::View * view = _viewer -> getView(index);
    osgGA::TrackballManipulator * manipulator = dynamic_cast<osgGA::TrackballManipulator *>(view -> getCameraManipulator());
    manipulator -> setTransformation(eye, center, up);
}

void
MorphologyViewerWidget::roll(double angle, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    Quat rotation = Quat(-angle, look);
    Vec3f rotated_up = rotation * up;
    rotated_up.normalize();
    _set_transformation(index, eye, center, rotated_up);

}

void
MorphologyViewerWidget::pitch(double angle, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    Quat rotation = Quat(-angle, side);
    Vec3f rotated_up = rotation * up;
    rotated_up.normalize();
    Vec3f rotated_look = rotation * look;
    rotated_look.normalize();
    _set_transformation(index, center - rotated_look * distance ,center, rotated_up);
}

void
MorphologyViewerWidget::yaw(double angle, unsigned int index)
{
    osg::Vec3d eye;
    osg::Vec3d center;
    osg::Vec3d up;
    osg::Vec3d look;
    osg::Vec3d side;
    double distance;
    _get_transformation(index, eye, center, distance, up, look, side);
    Quat rotation = Quat(-angle, up);
    Vec3f rotated_look = rotation * look;
    rotated_look.normalize();
    _set_transformation(index, center - rotated_look * distance ,center, up);
}


void
MorphologyViewerWidget::keyReleaseEvent( QKeyEvent* event )
{
  QString keyString   = event->text();
  const char* keyData = keyString.toLocal8Bit().data();

  this->getEventQueue()->keyRelease( osgGA::GUIEventAdapter::KeySymbol( *keyData ) );
}

void
MorphologyViewerWidget::mouseMoveEvent( QMouseEvent* event )
{
    this->getEventQueue()->mouseMotion( static_cast<float>( event->x() )
                                      , static_cast<float>( event->y() )
                                      );
}

void
MorphologyViewerWidget::mousePressEvent( QMouseEvent* event )
{
    // 1 = left mouse button
    // 2 = middle mouse button
    // 3 = right mouse button

    unsigned int button = 0;

    switch( event->button() )
    {
        case Qt::LeftButton:    button = 1;
                                break;

        case Qt::MiddleButton:  button = 2;
                                break;

        case Qt::RightButton:   button = 3;
                                break;

        default:                break;
    }

    this->getEventQueue()->mouseButtonPress( static_cast<float>( event->x() )
                                           , static_cast<float>( event->y() )
                                           , button
                                           );
}

void
MorphologyViewerWidget::mouseReleaseEvent(QMouseEvent* event)
{

    // 1 = left mouse button
    // 2 = middle mouse button
    // 3 = right mouse button

    unsigned int button = 0;

    switch( event->button() )
    {
        case Qt::LeftButton:    button = 1;
                                break;

        case Qt::MiddleButton:  button = 2;
                                break;

        case Qt::RightButton:   button = 3;
                                break;

        default:                break;
    }

    this->getEventQueue()->mouseButtonRelease( static_cast<float>( event->x() )
                                           , static_cast<float>( event->y() )
                                           , button
                                           );

}

void
MorphologyViewerWidget::wheelEvent( QWheelEvent* event )
{

    event->accept();
    int delta = event->delta();

    osgGA::GUIEventAdapter::ScrollingMotion motion = delta > 0 ?    osgGA::GUIEventAdapter::SCROLL_UP
                                                               :    osgGA::GUIEventAdapter::SCROLL_DOWN;

    this->getEventQueue()->mouseScroll( motion );
}

bool
MorphologyViewerWidget::event( QEvent* event )
{
    bool handled = QGLWidget::event( event );

    // This ensures that the OSG widget is always going to be repainted after the
    // user performed some interaction. Doing this in the event handler ensures
    // that we don't forget about some event and prevents duplicate code.
    switch( event->type() )
    {
        case QEvent::KeyPress:
        case QEvent::KeyRelease:
        case QEvent::MouseButtonDblClick:
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonRelease:
        case QEvent::MouseMove:
        case QEvent::Wheel:                 this->update();
                                            break;
        default:                            break;
    }

    return( handled );
}

void
MorphologyViewerWidget::onHome()
{
    osgViewer::ViewerBase::Views views;
    _viewer->getViews( views );

    for( std::size_t i = 0; i < views.size(); i++ )
    {
        osgViewer::View* view = views.at(i);
        view->home();
    }
}

void
MorphologyViewerWidget::onResize( int width, int height )
{
    // std::vector<osg::Camera*> cameras;
    // _viewer->getCameras( cameras );
    // float width_factor = static_cast<float>(width)
    //                    / static_cast<float>(_previous_width);
    // float height_factor = static_cast<float>(height)
    //                     / static_cast<float>(_previous_height);
    // for(osg::Camera * camera : cameras)
    // {
    //     osg::Viewport * viewport = camera -> getViewport();
    //     cout << "x => " << viewport -> x()      * width_factor << endl;
    //     cout << "y => " << viewport -> y()      * height_factor << endl;
    //     camera -> setViewport( viewport -> x()      * width_factor
    //                          , viewport -> y()      * height_factor
    //                          , viewport -> width()  * width_factor
    //                          , viewport -> height() * height_factor
    //                          );
    //     camera -> setProjectionMatrixAsPerspective( 30.f
    //                                               , (viewport -> width()  * width_factor)
    //                                               / (viewport -> height() * height_factor)
    //                                               , 1.0f
    //                                               , 10000.0f
    //                                               );
    // }
    _previous_height = height;
    _previous_width  = width;
}

osgGA::EventQueue*
MorphologyViewerWidget::getEventQueue() const
{
    osgGA::EventQueue* event_queue = _graphics_window -> getEventQueue();

    if( event_queue ) return( event_queue );
    else              throw( std::runtime_error( "Unable to obtain valid event queue") );
}
