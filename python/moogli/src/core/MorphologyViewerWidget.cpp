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
{
    float aspect_ratio = static_cast<float>( this->width()) / static_cast<float>( this->height() );

    osg::StateSet* stateSet = morphology -> get_scene_graph()->getOrCreateStateSet();
    osg::Material* material = new osg::Material;

    material->setColorMode( osg::Material::AMBIENT_AND_DIFFUSE );

    stateSet->setAttributeAndModes( material, osg::StateAttribute::ON );
    stateSet->setMode( GL_DEPTH_TEST, osg::StateAttribute::ON );

    osg::Camera* camera = new osg::Camera;
    camera->setViewport( 0, 0, this->width(), this->height() );
    camera->setClearColor( osg::Vec4( 0.0f, 0.0f, 0.0f, 1.f ) );
    camera->setProjectionMatrixAsPerspective( 30.f, aspect_ratio, 1.0f, 10000.0f );
    camera->setGraphicsContext( _graphics_window );
    osg::StateSet* stateset = camera -> getOrCreateStateSet();
    stateset->setGlobalDefaults();
    osgViewer::View* view = new osgViewer::View;
    view->setCamera( camera );
    view->setSceneData( morphology -> get_scene_graph() );
    view->addEventHandler( new osgViewer::StatsHandler );
    _manipulator = new osgGA::TrackballManipulator;
    view->setCameraManipulator( _manipulator );
    view -> addEventHandler(new KeyboardHandler());
    _viewer->addView( view );
    _viewer->setThreadingModel( osgViewer::CompositeViewer::SingleThreaded );

    // This ensures that the widget will receive keyboard events. This focus
    // policy is not set by default. The default, Qt::NoFocus, will result in
    // keyboard events that are ignored.
    this->setFocusPolicy( Qt::StrongFocus );
    this->setMinimumSize( 10
                        , 10
                        );

    // Ensures that the widget receives mouse move events even though no
    // mouse button has been pressed. We require this in order to let the
    // graphics window switch viewports properly.
    this->setMouseTracking( true );
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

void
MorphologyViewerWidget::keyPressEvent( QKeyEvent* event )
{
    QString keyString   = event->text();
    const char* keyData = keyString.toLocal8Bit().data();

    if( event->key() == Qt::Key_A )   { }
    else if( event->key() == Qt::Key_Space ) { this->onHome(); }
    else if( event->key() == Qt::Key_Up )
    {
        // osg::Matrix view_matrix = viewer.getCamera()->getViewMatrix();
        // view_matrix.getLookAt(eye,centre,up);
    // osg::Vec3d translation = view_matrix.getTrans();
    // translation.y() = translation.y() + translate_up_factor;
    // view_matrix.setTrans(translation);
    // viewer.getCamera()->setViewMatrix(view_matrix);
        osg::Vec3d eye;
        osg::Vec3d center;
        osg::Vec3d up;
        _manipulator -> getTransformation(eye, center, up);
        up.normalize();
        const osg::BoundingSphere& bs = morphology -> get_scene_graph() -> getBound();
        osg::Vec3d factor = up * (bs.radius() * 0.05);
        _manipulator -> setTransformation(eye + factor, center + factor, up);
        // morphology -> get_scene_graph() -> getMatrix().getLookAt(eye, center, up);

        // morphology -> get_scene_graph() -> preMult(osg::Matrix::translate(up.x(), up.y(), up.z()));
        //preMult(osg::Matrix::translate(0, -150, 0));
        // this->getEventQueue()->keyPress( osgGA::GUIEventAdapter::KEY_R );
    }

    else if( event->key() == Qt::Key_Down )
    {
        // osg::Matrix view_matrix = viewer.getCamera()->getViewMatrix();
        // view_matrix.getLookAt(eye,centre,up);
    // osg::Vec3d translation = view_matrix.getTrans();
    // translation.y() = translation.y() + translate_up_factor;
    // view_matrix.setTrans(translation);
    // viewer.getCamera()->setViewMatrix(view_matrix);
        osg::Vec3d eye;
        osg::Vec3d center;
        osg::Vec3d up;
        _manipulator -> getTransformation(eye, center, up);
        up.normalize();
        const osg::BoundingSphere& bs = morphology -> get_scene_graph() -> getBound();
        osg::Vec3d factor = up * (bs.radius() * 0.05);
        _manipulator -> setTransformation(eye - factor, center - factor, up);
        // morphology -> get_scene_graph() -> getMatrix().getLookAt(eye, center, up);

        // morphology -> get_scene_graph() -> preMult(osg::Matrix::translate(up.x(), up.y(), up.z()));
        //preMult(osg::Matrix::translate(0, -150, 0));
        // this->getEventQueue()->keyPress( osgGA::GUIEventAdapter::KEY_R );
    }
    else if( event->key() == Qt::Key_Left )
    {
        osg::Vec3d eye;
        osg::Vec3d center;
        osg::Vec3d up;
        _manipulator -> getTransformation(eye, center, up);
        up.normalize();
        osg::Vec3d look_vector = center - eye;
        look_vector.normalize();
        osg::Vec3d right_side = look_vector ^ up;
        right_side.normalize();
        const osg::BoundingSphere& bs = morphology -> get_scene_graph() -> getBound();
        osg::Vec3d factor = right_side * (bs.radius() * 0.05);
        _manipulator -> setTransformation(eye - factor, center - factor, up);
    }

    else if( event->key() == Qt::Key_Right )
    {
        osg::Vec3d eye;
        osg::Vec3d center;
        osg::Vec3d up;
        _manipulator -> getTransformation(eye, center, up);
        up.normalize();
        osg::Vec3d look_vector = center - eye;
        look_vector.normalize();
        osg::Vec3d right_side = look_vector ^ up;
        right_side.normalize();
        const osg::BoundingSphere& bs = morphology -> get_scene_graph() -> getBound();
        osg::Vec3d factor = right_side * (bs.radius() * 0.05);
        _manipulator -> setTransformation(eye + factor, center + factor, up);
    }
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
    std::vector<osg::Camera*> cameras;
    _viewer->getCameras( cameras );
    cameras[0]->setViewport( 0, 0, this->width(), this->height() );
}

osgGA::EventQueue*
MorphologyViewerWidget::getEventQueue() const
{
    osgGA::EventQueue* event_queue = _graphics_window -> getEventQueue();

    if( event_queue ) return( event_queue );
    else              throw( std::runtime_error( "Unable to obtain valid event queue") );
}
