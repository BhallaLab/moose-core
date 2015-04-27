%Module viewer

%Import QtGui/QtGuimod.sip

class Morphology;

class MorphologyViewer : public QWidget
{

Q_OBJECT

%TypeHeaderCode
#include "includes.hpp"
#include "core/Morphology.hpp"
#include "core/Selector.hpp"
%End

public:

    MorphologyViewer( Morphology * morphology         = 0
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

signals:
    void compartment_dragged(const QString &compartment_id);

};
