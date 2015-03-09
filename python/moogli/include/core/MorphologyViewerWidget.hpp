#ifndef _MORPHOLOGY_VIEWER_WIDGET_HPP_
#define _MORPHOLOGY_VIEWER_WIDGET_HPP_

#include "includes.hpp"
#include "core/Morphology.hpp"
#include "core/Selector.hpp"
#include "KeyboardHandler.hpp"
#include "core/SelectInfo.hpp"
#include <chrono>
using namespace std;


class MorphologyViewerWidget : public QGLWidget
{
  Q_OBJECT

public:
    Morphology * morphology;
    MorphologyViewerWidget( Morphology * morphology
                          , QWidget * parent             = 0
                          , const QGLWidget* shareWidget = 0
                          , Qt::WindowFlags f            = 0
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

private:

    virtual void
    onHome();

    virtual void
    onResize(int width, int height);

    osgViewer::View *
    createView();

    osg::Camera *
    createCamera();

    osgGA::EventQueue*
    getEventQueue() const;

    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> _graphics_window;
    osg::ref_ptr<osgViewer::CompositeViewer> _viewer;
    osgGA::TrackballManipulator * _manipulator;
};

#endif /* _MORPHOLOGY_VIEWER_WIDGET_HPP_ */
