%Module _moogli

%Import QtGui/QtGuimod.sip
%Import QtOpenGL/QtOpenGLmod.sip
class SelectInfo
{
%TypeHeaderCode
#include "core/SelectInfo.hpp"
%End
public:
    SelectInfo();
    const char * get_id();
    int get_event_type();
    void set_id(const char * id);
    void set_event_type(int event_type);
};

class Morphology
{

%TypeHeaderCode
#include "includes.hpp"
#include "core/Compartment.hpp"
#include "core/Morphology.hpp"
%End

public:

    Morphology( const char *        name               = ""
              , const unsigned int  lod_resolution     = 3
              , const float         lod_distance_delta = 50.0f
              , const unsigned int  min_points         = 10
              , const unsigned int  points_delta       = 2
              );

    int
    add_compartment( const char *    compartment_id
                   , const char *    neuron_id
                   , double          proximal_x
                   , double          proximal_y
                   , double          proximal_z
                   , double          proximal_d
                   , double          distal_x
                   , double          distal_y
                   , double          distal_z
                   , double          distal_d
                   );

    PyObject *
    set_compartment_order(PyObject * compartment_order);

    void
    set_initial_color(float r, float g, float b, float a);

    void
    set_final_color(float r, float g, float b, float a);

    void
    set_membrane_voltages(PyObject * vms);

    void
    set_base_membrane_voltage(double base_vm);

    void
    set_peak_membrane_voltage(double peak_vm);

    void
    destroy_group(const char * group_id);

    void
    modify_group( const char *  group_id
                , PyObject *    compartment_ids
                , double        base_value
                , double        peak_value
                , PyObject *    base_color
                , PyObject *    peak_color
                );

    void
    create_group( const char *  group_id
                , PyObject *    compartment_ids
                , double        base_value
                , double        peak_value
                , PyObject *    base_color
                , PyObject *    peak_color
                );

    void
    set_color( const char * group_id
             , PyObject   * values
             );

private:

    Morphology(const Morphology &);
};

class MorphologyViewer : public QWidget
{

%TypeHeaderCode
#include "includes.hpp"
#include "core/Morphology.hpp"
#include "core/Selector.hpp"
#include "core/MorphologyViewer.hpp"
%End

public:
    SelectInfo * select_info;

    MorphologyViewer( Morphology * morphology /TransferThis/    = 0
                    , int width                       = 1000
                    , int height                      = 1000
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
    frame();

    void
    set_background_color(float r, float g, float b, float a);

private:
    MorphologyViewer(const MorphologyViewer &);

signals:
    void compartment_dragged(const QString &compartment_id);

};

class MorphologyViewerWidget : public QGLWidget
{
%TypeHeaderCode
#include "includes.hpp"
#include "core/Morphology.hpp"
#include "core/MorphologyViewerWidget.hpp"
%End

public:

    double up_distance;
    double down_distance;
    double left_distance;
    double right_distance;
    double forward_distance;
    double backward_distance;
    double zoom_factor;
    double roll_angle;
    double pitch_angle;
    double yaw_angle;

    MorphologyViewerWidget( Morphology * morphology /Transfer/
                         , QWidget * parent             = 0
                         , const QGLWidget* shareWidget = 0
                         , Qt::WindowFlags f            = 0
                         );

    Morphology *
    get_morphology();

    void
    add_view( int x
            , int y
            , int width
            , int height
            );

    void
    split_horizontally( unsigned int view_index   = 0
                      , unsigned int width_factor = 2
                      );

    void
    split_vertically( unsigned int view_index    = 0
                    , unsigned int height_factor = 2
                    );
    void
    home(unsigned int index = 0);

    void
    forward( double distance
           , unsigned int index = 0
           );
    void
    backward( double distance
            , unsigned int index = 0
            );
    void
    left( double distance
        , unsigned int index = 0
        );
    void
    right( double distance
         , unsigned int index = 0
         );

    void
    up( double distance
      , unsigned int index = 0
      );

    void
    down( double distance
        , unsigned int index = 0
        );

    void
    zoom( double factor
        , unsigned int index = 0
        );

    void
    roll( double angle
        , unsigned int index = 0
        );

    void
    pitch( double angle
         , unsigned int index = 0
         );

    void
    yaw( double angle
       , unsigned int index = 0
       );

    virtual
    ~MorphologyViewerWidget();

protected:

    virtual void paintEvent( QPaintEvent* paintEvent );
    virtual void paintGL();
    virtual void resizeGL( int width, int height );

    virtual void keyPressEvent( QKeyEvent* event );
    virtual void keyReleaseEvent( QKeyEvent* event );

    virtual void mouseMoveEvent( QMouseEvent* event );
    virtual void mousePressEvent( QMouseEvent* event );
    virtual void mouseReleaseEvent( QMouseEvent* event );
    virtual void wheelEvent( QWheelEvent* event );

    virtual bool event( QEvent* event );
};
