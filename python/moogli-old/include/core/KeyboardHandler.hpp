#ifndef __KEYBOARD_HANDLER_HPP__
#define __KEYBOARD_HANDLER_HPP__

#include "includes.hpp"

using namespace std;
using namespace osg;

class KeyboardHandler : public osgGA::GUIEventHandler
{
public:

    constexpr static double translate_up_factor      = 20.0;
    constexpr static double translate_down_factor    = 20.0;
    constexpr static double translate_left_factor    = 20.0;
    constexpr static double translate_right_factor   = 20.0;
    constexpr static double translate_forward_factor = 10.0;
    constexpr static double translate_backward_factor= 20.0;
    constexpr static double clockwise_roll_factor    = 5 * 3.14 / 180;
    constexpr static double clockwise_pitch_factor    = 5 * 3.14 / 180;
    constexpr static double clockwise_yaw_factor    = 5 * 3.14 / 180;
    constexpr static double counterclockwise_roll_factor    = 5 * 3.14 / 180;
    constexpr static double counterclockwise_pitch_factor    = 5 * 3.14 / 180;
    constexpr static double counterclockwise_yaw_factor    = 5 * 3.14 / 180;


    KeyboardHandler() {}
    ~KeyboardHandler() {}
    bool
    handle( const osgGA::GUIEventAdapter& ea
          ,osgGA::GUIActionAdapter& aa
          );
    void
    translate_up(osgViewer::Viewer & viewer);

    void
    translate_down(osgViewer::Viewer & viewer);

    void
    translate_left(osgViewer::Viewer & viewer);

    void
    translate_right(osgViewer::Viewer & viewer);

    void
    translate_forward(osgViewer::Viewer & viewer);

    void
    translate_backward(osgViewer::Viewer & viewer);

    void
    clockwise_roll(osgViewer::Viewer & viewer);

    void
    counterclockwise_roll(osgViewer::Viewer & viewer);

    void
    clockwise_pitch(osgViewer::Viewer & viewer);

    void
    counterclockwise_pitch(osgViewer::Viewer & viewer);

    void
    clockwise_yaw(osgViewer::Viewer & viewer);

    void
    counterclockwise_yaw(osgViewer::Viewer & viewer);

};

#endif /* __KEYBOARD_HANDLER_HPP__ */
