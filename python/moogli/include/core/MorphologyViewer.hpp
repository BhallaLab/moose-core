#ifndef __MORPHOLOGY_VIEWER_HPP__
#define __MORPHOLOGY_VIEWER_HPP__

#include "includes.hpp"
#include "core/Morphology.hpp"
#include "core/PickHandler.hpp"
#include "core/Selector.hpp"
#include "core/SelectInfo.hpp"

using namespace std;

class MorphologyViewer : public QWidget
{
    Q_OBJECT
public:

    Morphology *  _morphology;
    const int width;
    const int height;
    const double fovy;
    const double z_near;
    const double z_far;
    const Vec4 clear_color;
    const bool  stats_handler;
    const bool  pick_handler;
    const bool  capture_handler;
    const int single_capture_key;
    const int continuous_capture_key;
    const string capture_location;
    const string capture_format;
    const string window_name;
    const bool window_decoration;
    SelectInfo * select_info;

    MorphologyViewer( const char * name
                    , int width
                    , int height
                    , double fovy                     = 30.0f
                    , double z_near                   = 1.0f
                    , double z_far                    = 10000.0f
                    , double r                        = 0.0
                    , double g                        = 0.0
                    , double b                        = 0.0
                    , double a                        = 1.0
                    , bool  stats_handler             = true
                    , bool  pick_handler              = true
                    , bool  capture_handler           = true
                    , int single_capture_key          = 'w'
                    , int continuous_capture_key      = 'W'
                    , const char * capture_location   = "./"
                    , const char * capture_format     = "jpeg"
                    , const char * window_name        = "Moogli"
                    , int window_decoration           = false
                    );

    MorphologyViewer( Morphology * morphology
                    , int width
                    , int height
                    , double fovy                     = 30.0f
                    , double z_near                   = 1.0f
                    , double z_far                    = 10000.0f
                    , double r                        = 0.0
                    , double g                        = 0.0
                    , double b                        = 0.0
                    , double a                        = 1.0
                    , bool  stats_handler             = true
                    , bool  pick_handler              = true
                    , bool  capture_handler           = true
                    , int single_capture_key          = 'w'
                    , int continuous_capture_key      = 'W'
                    , const char * capture_location   = "./"
                    , const char * capture_format     = "jpeg"
                    , const char * window_name        = "Moogli"
                    , bool window_decoration          = false
                    );
    void
    create_view();

    QWidget*
    create_graphics_widget();

    Camera *
    create_camera();

    void
    frame()
    {
        _viewer.frame();
    }

    void
    set_background_color(float r, float g, float b, float a);

signals:
    void compartment_dragged(const QString &compartment_id);


protected:

    virtual void
    paintEvent( QPaintEvent* event )
    {
        _viewer.frame();
    }

    void
    emit_signal(const string & compartment_id);

private:

    moose_id_collection_t selections;

    MorphologyViewer(const MorphologyViewer &);

    osgViewer::StatsHandler *
    _get_stats_handler();

    osgViewer::ScreenCaptureHandler *
    _get_capture_handler( const string & capture_location
                        , const string & capture_format
                        , int single_capture_key
                        , int continuous_capture_key
                        );

    QTimer              _timer;
    unsigned int        _view_count;
    QGridLayout  *      _grid_layout;
    osgViewer::Viewer   _viewer;
};



#endif  /*  __MORPHOLOGY_VIEWER_HPP__ */
