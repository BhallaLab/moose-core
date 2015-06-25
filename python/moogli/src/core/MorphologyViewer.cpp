#include "core/MorphologyViewer.hpp"
#include <QtOpenGL/QGLFormat>
MorphologyViewer::MorphologyViewer( const char * name
                                  , int width
                                  , int height
                                  , double fovy
                                  , double z_near
                                  , double z_far
                                  , double r
                                  , double g
                                  , double b
                                  , double a
                                  , bool  stats_handler
                                  , bool  pick_handler
                                  , bool  capture_handler
                                  , int single_capture_key
                                  , int continuous_capture_key
                                  , const char * capture_location
                                  , const char * capture_format
                                  , const char * window_name
                                  , int window_decoration
                                  ) : QWidget()
                                    , _morphology(new Morphology(name))
                                    , width(width)
                                    , height(height)
                                    , fovy(fovy)
                                    , z_near(z_near)
                                    , z_far(z_far)
                                    , clear_color(Vec4(r,g,b,a))
                                    , stats_handler(stats_handler)
                                    , pick_handler(pick_handler)
                                    , capture_handler(capture_handler)
                                    , single_capture_key(single_capture_key)
                                    , continuous_capture_key(continuous_capture_key)
                                    , capture_location(capture_location)
                                    , capture_format(capture_format)
                                    , window_name(window_name)
                                    , window_decoration(window_decoration)
{
    mode = MOUSE;
    select_info  = new SelectInfo();
    _grid_layout = new QGridLayout();
    _grid_layout->setContentsMargins(0,0,0,0);
    setLayout(_grid_layout);
    setup_toolbar();
    create_view();
    setStyleSheet("background-color:black;");
    connect( &_timer, SIGNAL(timeout()), this, SLOT(update()) );
    _timer.start( 0 );
}

MorphologyViewer::MorphologyViewer( const MorphologyViewer & morphologyViewer
                                  ) : QWidget()
                                    , _morphology(morphologyViewer._morphology)
                                    , width(morphologyViewer.width)
                                    , height(morphologyViewer.height)
                                    , fovy(morphologyViewer.fovy)
                                    , z_near(morphologyViewer.z_near)
                                    , z_far(morphologyViewer.z_far)
                                    , clear_color(morphologyViewer.clear_color)
                                    , stats_handler(morphologyViewer.stats_handler)
                                    , pick_handler(morphologyViewer.pick_handler)
                                    , capture_handler(morphologyViewer.capture_handler)
                                    , single_capture_key(morphologyViewer.single_capture_key)
                                    , continuous_capture_key(morphologyViewer.continuous_capture_key)
                                    , capture_location(morphologyViewer.capture_location)
                                    , capture_format(morphologyViewer.capture_format)
                                    , window_name(morphologyViewer.window_name)
                                    , window_decoration(morphologyViewer.window_decoration)
                                    , select_info(morphologyViewer.select_info)
{ }


MorphologyViewer::MorphologyViewer( Morphology * morphology
                                  , int width
                                  , int height
                                  , double fovy
                                  , double z_near
                                  , double z_far
                                  , double r
                                  , double g
                                  , double b
                                  , double a
                                  , bool  stats_handler
                                  , bool  pick_handler
                                  , bool  capture_handler
                                  , int single_capture_key
                                  , int continuous_capture_key
                                  , const char * capture_location
                                  , const char * capture_format
                                  , const char * window_name
                                  , bool window_decoration
                                  ) : QWidget()
                                    , _morphology(morphology)
                                    , width(width)
                                    , height(height)
                                    , fovy(fovy)
                                    , z_near(z_near)
                                    , z_far(z_far)
                                    , clear_color(Vec4(r,g,b,a))
                                    , stats_handler(stats_handler)
                                    , pick_handler(pick_handler)
                                    , capture_handler(capture_handler)
                                    , single_capture_key(single_capture_key)
                                    , continuous_capture_key(continuous_capture_key)
                                    , capture_location(capture_location)
                                    , capture_format(capture_format)
                                    , window_name(window_name)
                                    , window_decoration(window_decoration)
{
    mode = MOUSE;
    select_info  = new SelectInfo();
    _grid_layout = new QGridLayout();
    _grid_layout->setContentsMargins(0,0,0,0);
    setLayout(_grid_layout);
    setup_toolbar();
    create_view();
    // setStyleSheet("background-color:black;");
    // connect( &_timer, SIGNAL(timeout()), this, SLOT(update()) );
    // _timer.start( 0 );

}

void
MorphologyViewer::handle_translate_positive_x()
{
    std::cerr << "Zooming out!" << std::endl;

    osg::Vec3f eye      =   osg::Vec3f(0.0,-200.0, 0.0);
    osg::Vec3f centre   =   osg::Vec3f(0.0, 0.0  , 0.0);
    osg::Vec3f up       =   osg::Vec3f(0.0, 0.0  , 1.0);

    const osg::Matrix matrix =_viewer.getCamera()->getViewMatrix();
    matrix.getLookAt(eye,centre,up);
    osg::Vec3f look_vector = centre - eye;
    //look_vector.normalize();
    look_vector = look_vector / (look_vector.length());
    eye = eye - (look_vector * 30.0f);
    _viewer.getCamera()->setViewMatrixAsLookAt(eye, centre, up);
    _viewer.frame();

    // auto matrix = camera -> getViewMatrix();
    // auto trans  = matrix.getTrans();
    // trans.z() = trans.z() - 0.5;
    // matrix.setTrans(trans);
    // manipulator -> setByMatrix( osg::Matrixd::translate( manipulator -> getCenter()
    //                                                    + osg::Vec3d(0.05, 0.0, 0.0)
    //                                                    )
    //                           * osg::Matrixd::rotate(manipulator -> getMatrix().getRotate())
    //                           );
    // const osg::Matrix & matrix = _morphology -> get_scene_graph() -> getMAtrix()
    // _morphology -> get_scene_graph() -> setPosition( _morphology -> get_scene_graph() -> getPosition()
    //                                                , osg::Vec3d(0.0, 0.0, -0.5)
    //                                                );
}


void
MorphologyViewer::setup_toolbar()
{
    _toolbar = new QToolBar();
    _toggle_mode_button = new QPushButton();
    _toggle_mode_button -> setText("Keyboard Mode");
    // _toggle_mode_button -> setStyleSheet(QString('QPushButton {background-color: black; color: white;}'));
    connect( _toggle_mode_button, SIGNAL(released())
           ,this, SLOT(toggle_mode())
           );

    // _zoom_in = new QPushButton();
    // _zoom_in -> setText("Zoom In");

    // _zoom_out = new QPushButton();
    // _zoom_out -> setText("Zoom Out");

    // _translate_positive_x = new QPushButton();
    // _translate_positive_x -> setText("M +x");
    // connect( _translate_positive_x, SIGNAL(released())
    //        ,this, SLOT(handle_translate_positive_x())
    //        );
    // _translate_positive_y = new QPushButton();
    // _translate_positive_y -> setText("M +y");

    // _translate_positive_z = new QPushButton();
    // _translate_positive_z -> setText("M +z");

    // _translate_negative_x = new QPushButton();
    // _translate_negative_x -> setText("M -x");

    // _translate_negative_y = new QPushButton();
    // _translate_negative_y -> setText("M -y");

    // _translate_negative_z = new QPushButton();
    // _translate_negative_z -> setText("M -z");

    // _rotate_clockwise_x = new QPushButton();
    // _rotate_clockwise_x -> setText("C x");

    // _rotate_clockwise_y = new QPushButton();
    // _rotate_clockwise_y -> setText("C y");

    // _rotate_clockwise_z = new QPushButton();
    // _rotate_clockwise_z -> setText("C z");

    // _rotate_counterclockwise_x = new QPushButton();
    // _rotate_counterclockwise_x -> setText("CC x");

    // _rotate_counterclockwise_y = new QPushButton();
    // _rotate_counterclockwise_y -> setText("CC y");

    // _rotate_counterclockwise_z = new QPushButton();
    // _rotate_counterclockwise_z -> setText("CC z");

    //Uncomment this to see the button
    _toolbar -> addWidget(_toggle_mode_button);


    // _toolbar -> setStyleSheet("background-color:black;");
    // _toolbar -> addWidget(_zoom_out);
    // _toolbar -> addWidget(_translate_positive_x);
    // _toolbar -> addWidget(_translate_positive_y);
    // _toolbar -> addWidget(_translate_positive_z);
    // _toolbar -> addWidget(_translate_negative_x);
    // _toolbar -> addWidget(_translate_negative_y);
    // _toolbar -> addWidget(_translate_negative_z);
    // _toolbar -> addWidget(_rotate_clockwise_x);
    // _toolbar -> addWidget(_rotate_clockwise_y);
    // _toolbar -> addWidget(_rotate_clockwise_z);
    // _toolbar -> addWidget(_rotate_counterclockwise_x);
    // _toolbar -> addWidget(_rotate_counterclockwise_y);
    // _toolbar -> addWidget(_rotate_counterclockwise_z);

    // Uncomment this line to add toolbar
    //_grid_layout -> addWidget(_toolbar, 0, 0);
}

void
MorphologyViewer::create_view()
{
    QScrollArea* _scroll_area = new QScrollArea;
    _scroll_area -> setBackgroundRole(QPalette::Dark);
    _scroll_area -> setWidget(create_graphics_widget());
    _grid_layout -> addWidget(_scroll_area, 0, 0);
}

QWidget*
MorphologyViewer::create_graphics_widget()
{
    camera = create_camera();

    if(pick_handler)
    {
        Selector * selector = new Selector(_morphology -> get_scene_graph());
        selector -> select_info = select_info;
        _viewer.addEventHandler(selector);
    }

    if(capture_handler)
    {
        _viewer.addEventHandler(_get_capture_handler( capture_location
                                                    , capture_format
                                                    , single_capture_key
                                                    , continuous_capture_key
                                                    )
                               );
    }

    if(stats_handler)
    {
        _viewer.addEventHandler(_get_stats_handler()
                               );
    }

    _viewer.setCamera(camera);
    _viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);
    _viewer.setSceneData(_morphology -> get_scene_graph().get());

    // const osg::BoundingSphere& bs = _morphology -> get_scene_graph() -> getBound();
    // RECORD_INFO("x =>" + to_string(bs.center().x()));
    // RECORD_INFO("y =>" + to_string(bs.center().y()));
    // RECORD_INFO("z =>" + to_string(bs.center().z()));
    // RECORD_INFO("r =>" + to_string(bs.radius()));

    // _viewer.getCamera() -> setViewMatrixAsLookAt(bs.center()+osg::Vec3(0.0f, -(3*bs.radius()),0.0f), bs.center(), osg::Z_AXIS) ;

    keyboard_handler = new KeyboardHandler();
    // _viewer.addEventHandler(keyboard_handler);
    manipulator = new osgGA::TrackballManipulator();
    // osgGA::StandardManipulator::UPDATE_MODEL_SIZE
    //                                         | osgGA::StandardManipulator::COMPUTE_HOME_USING_BBOX
    //                                         | osgGA::StandardManipulator::PROCESS_MOUSE_WHEEL
    //                                         | osgGA::StandardManipulator::SET_CENTER_ON_WHEEL_FORWARD_MOVEMENT
    //                                         );
    // osg::Vec3d eye, center, up;
    // manipulator -> setHomePosition(eye, ,up)
     // osgGA::StandardManipulator::UPDATE_MODEL_SIZE
     //                                    | osgGA::StandardManipulator::COMPUTE_HOME_USING_BBOX
     //                                    | osgGA::StandardManipulator::PROCESS_MOUSE_WHEEL
     //                                    | osgGA::StandardManipulator::SET_CENTER_ON_WHEEL_FORWARD_MOVEMENT
     //                                    );

    _viewer.setCameraManipulator( manipulator);
    // manipulator -> setVerticalAxisFixed(true);
    // _viewer.setCameraManipulator( manipulator );
    // manipulator -> setAutoComputeHomePosition(true);
    // manipulator -> computeHomePosition(camera, true);
    // manipulator -> home(0.0);
    // manipulator -> updateCamera(* camera);
    _viewer.addEventHandler(new osgViewer::WindowSizeHandler());
    _viewer.setQuitEventSetsDone(false);
    osgQt::GraphicsWindowQt* gw =
    (osgQt::GraphicsWindowQt*)(camera -> getGraphicsContext());
    gw -> getGLWidget() -> setForwardKeyEvents(true);
    // QGLFormat format;
    // QGLFormat::setDefaultFormat(format);
    // format.setSampleBuffers(true);
    // format.setAlpha(true);
    // format.setRgba(true);
//    format.setSamples(16);
    // gw -> getGLWidget() -> setFormat(format);

    return gw -> getGLWidget();
}

void
MorphologyViewer::toggle_mode()
{
    if(mode == KEYBOARD) { mode = MOUSE;     }
    else                 { mode = KEYBOARD;  }
    if(mode == KEYBOARD)
    {
        RECORD_INFO("Setting null manipulator");
        _viewer.setCameraManipulator(nullptr);
        RECORD_INFO("Set null manipulator");
        keyboard_handler = new KeyboardHandler();
        _viewer.addEventHandler(keyboard_handler);
        _toggle_mode_button -> setText("Mouse Mode");
    }
    else
    {
        manipulator = new osgGA::TrackballManipulator();

        osg::Vec3f eye;
        osg::Vec3f centre;
        osg::Vec3f up;

        osg::Matrix view_matrix = _viewer.getCamera()->getViewMatrix();
        view_matrix.getLookAt(eye,centre,up);

        // osgGA::StandardManipulator::UPDATE_MODEL_SIZE
        //                                     | osgGA::StandardManipulator::COMPUTE_HOME_USING_BBOX
        //                                     | osgGA::StandardManipulator::PROCESS_MOUSE_WHEEL
        //                                     | osgGA::StandardManipulator::SET_CENTER_ON_WHEEL_FORWARD_MOVEMENT
        //                                     );
        // // manipulator -> setAutoComputeHomePosition(true);
        // manipulator -> home(0.0);
        _viewer.setCameraManipulator( manipulator);
        manipulator -> setHomePosition(eye, centre, up);
        _viewer.removeEventHandler(keyboard_handler);
        _toggle_mode_button -> setText("Keyboard Mode");
    }
}


void
MorphologyViewer::set_background_color(float r, float g, float b, float a)
{
    _viewer.getCamera() -> setClearColor(Vec4(r, g, b, a));
}

Camera *
MorphologyViewer::create_camera()
{
    // osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits();

    // // traits->windowName      = window_name;
    // // traits->windowDecoration= window_decoration;
    // traits->x               = 0;
    // traits->y               = 0;
    // traits->width           = width;
    // traits->height          = height;
    // traits->doubleBuffer    = true;
    // traits->alpha           = ds -> getMinimumNumAlphaBits() + 8;
    // traits->stencil         = ds -> getMinimumNumStencilBits() + 8;
    // traits->sampleBuffers   = ds -> getMultiSamples() + 8;
    // traits->samples         = ds -> getNumMultiSamples() + 8;

    // unsigned int major, minor;

    // traits -> getContextVersion(major, minor);

    // RECORD_INFO("Using OpenGL Version - " + to_string(major) + "." + to_string(minor));
    osg::DisplaySettings* ds = osg::DisplaySettings::instance().get();
    osg::GraphicsContext::Traits  * traits = new osg::GraphicsContext::Traits();

    Camera * camera = new Camera();
    traits -> windowDecoration  = false;
    // traits -> x                 = 0;
    // traits -> y                 = 0;
    // traits -> width             = width;
    // traits -> height            = height;
    traits -> doubleBuffer      = true;

    // double h = ds->getScreenHeight();
    // double w = ds->getScreenWidth() ;
    // RECORD_INFO("Screen Width => " + to_string(w));
    // RECORD_INFO("Screen Height => " + to_string(h));
    // double distance = ds->getScreenDistance();
    // double vfov = osg::RadiansToDegrees(atan2(h/2.0f,distance)*2.0);
    // double aspect_ratio = w / h;


    RECORD_INFO("Width: " + to_string(width));
    RECORD_INFO("Height: " + to_string(height));
    traits -> x = 0;
    traits -> y = 0;
    traits -> width = width;
    traits -> height = height;
    // static_cast<int>(traits -> width / aspect_ratio);

    // traits -> width = int(traits -> height * aspect_ratio);
    // RECORD_INFO("Traits Width: " + to_string(traits -> width));
    camera -> setGraphicsContext(new osgQt::GraphicsWindowQt(traits));
    camera-> setClearColor(clear_color);
    osg::StateSet* stateset = camera -> getOrCreateStateSet();
    stateset->setGlobalDefaults();
    stateset -> setMode( GL_BLEND, StateAttribute::ON );
    camera->setCullingMode( CullSettings::NEAR_PLANE_CULLING
                          | CullSettings::FAR_PLANE_CULLING
                          | CullSettings::VIEW_FRUSTUM_CULLING
                          | CullSettings::SMALL_FEATURE_CULLING
                          );

    camera-> setViewport( new osg::Viewport( 0
                                           , 0
                                           , traits->width
                                           , traits->height
                                           )
                        );

    camera -> setProjectionMatrixAsPerspective( 30.0f
                                              , static_cast<double>(traits->width)/
                                                static_cast<double>(traits->height)
                                              , 0.5f
                                              , 20000.0f
                                              );



    // state_set -> setMode( GL_RESCALE_NORMAL, StateAttribute::ON );
    //4 _state_set -> setRenderingHint( StateSet::TRANSPARENT_BIN );

    // Enable depth test so that an opaque polygon will occlude a transparent one behind it.
    // stateset->setMode( GL_DEPTH_TEST, StateAttribute::ON );

    // Conversely, disable writing to depth buffer so that
    // a transparent polygon will allow polygons behind it to shine thru.
    // OSG renders transparent polygons after opaque ones.
    // Depth * depth = new Depth();
    // depth -> setWriteMask( true );
    // _state_set->setAttributeAndModes( depth, StateAttribute::ON );


    return camera;
}

osgViewer::StatsHandler *
MorphologyViewer::_get_stats_handler()
{
    return (new osgViewer::StatsHandler());
}

osgViewer::ScreenCaptureHandler *
MorphologyViewer::_get_capture_handler( const string & capture_location
                                      , const string & capture_format
                                      , int single_capture_key
                                      , int continuous_capture_key
                                      )
{
    auto* capture_operation =
        new osgViewer::ScreenCaptureHandler::WriteToFile( capture_location
                                                        , capture_format
                                                        );

    auto* capture_handler = new osgViewer::ScreenCaptureHandler(
        dynamic_cast<osgViewer::ScreenCaptureHandler::CaptureOperation *>(capture_operation)
                                                               );

    capture_handler -> setKeyEventTakeScreenShot(single_capture_key);
    capture_handler -> setKeyEventToggleContinuousCapture(continuous_capture_key);
    return capture_handler;
}

// void
// MorphologyViewer::emit_signal(const string & compartment_id)
// {
//     emit compartment_dragged(QString(compartment_id.c_str()));
// }

