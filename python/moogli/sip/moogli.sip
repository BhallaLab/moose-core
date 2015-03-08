%Module moogli

%Import QtGui/QtGuimod.sip

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
