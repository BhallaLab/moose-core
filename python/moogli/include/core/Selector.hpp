#ifndef __SELECTOR_HPP__
#define __SELECTOR_HPP__

#include "includes.hpp"
#include "core/SelectInfo.hpp"

using namespace std;
using namespace osg;

class MorphologyViewer;

class Selector : public osgGA::GUIEventHandler
{
public:
    QMenu * menu;
    int mode;
    // _neurons;
    // _compartments;
    SelectInfo * select_info;

    Selector();

    virtual bool handle( const osgGA::GUIEventAdapter& ea
                       , osgGA::GUIActionAdapter& aa
                       );

private:

    Geometry *
    _get_intersection( const osgGA::GUIEventAdapter& ea
                     , osgViewer::Viewer* viewer
                     );

    void
    _deselect();

    bool
    _select_compartment(Geometry * geometry);

    bool
    _select_neuron(Geometry * geometry);

};

#endif /* __SELECTOR_HPP__ */
