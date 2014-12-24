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
    select_info  = new SelectInfo();
    _grid_layout = new QGridLayout();
    _grid_layout->setContentsMargins(0,0,0,0);
    setLayout(_grid_layout);
    create_view();
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
    select_info  = new SelectInfo();
    _grid_layout = new QGridLayout();
    _grid_layout->setContentsMargins(0,0,0,0);
    setLayout(_grid_layout);
    create_view();
    // connect( &_timer, SIGNAL(timeout()), this, SLOT(update()) );
    // _timer.start( 0 );

}

void
MorphologyViewer::create_view()
{
    QScrollArea* _scroll_area = new QScrollArea;
    _scroll_area->setBackgroundRole(QPalette::Dark);
    _scroll_area->setWidget(create_graphics_widget());
    _grid_layout -> addWidget(_scroll_area, 0, 0);
}

QWidget*
MorphologyViewer::create_graphics_widget()
{
    osg::Camera* camera = create_camera();

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
    _viewer.setCameraManipulator( new osgGA::TrackballManipulator );
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
    osg::GraphicsContext::Traits  * traits = new osg::GraphicsContext::Traits(ds);

    Camera * camera = new Camera();

    double h = ds->getScreenHeight();
    double w = ds->getScreenWidth() ;
    double distance = ds->getScreenDistance();
    double vfov = osg::RadiansToDegrees(atan2(h/2.0f,distance)*2.0);
    double aspect_ratio = w/h;
    // RECORD_INFO("Width: " + to_string(width));
    // RECORD_INFO("Height: " + to_string(height));
    traits -> x = 0;
    traits -> y = 0;
    traits -> width = width;
    traits -> height = static_cast<int>(traits -> width / aspect_ratio);
    // traits -> width = int(traits -> height * aspect_ratio);
    // RECORD_INFO("Traits Width: " + to_string(traits -> width));
    camera -> setProjectionMatrixAsPerspective( vfov
                                              , aspect_ratio
                                              , z_near
                                              , z_far
                                              );


    camera -> setGraphicsContext(new osgQt::GraphicsWindowQt(traits));
    camera-> setClearColor(clear_color);
    camera-> setViewport( new osg::Viewport( 0
                                           , 0
                                           , traits->width
                                           , traits->height
                                           )
                        );

    osg::StateSet* stateset = camera -> getOrCreateStateSet();
    stateset->setGlobalDefaults();
    stateset -> setMode( GL_BLEND, StateAttribute::ON );
    camera->setCullingMode( CullSettings::NEAR_PLANE_CULLING
                          | CullSettings::FAR_PLANE_CULLING
                          | CullSettings::VIEW_FRUSTUM_CULLING
                          | CullSettings::SMALL_FEATURE_CULLING
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

void
MorphologyViewer::emit_signal(const string & compartment_id)
{
    emit compartment_dragged(QString(compartment_id.c_str()));
}

