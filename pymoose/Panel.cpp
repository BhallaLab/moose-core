#ifndef _pymoose_Panel_cpp
#define _pymoose_Panel_cpp
#include "Panel.h"
using namespace pymoose;
const std::string Panel::className_ = "Panel";
Panel::Panel(Id id):PyMooseBase(id){}
Panel::Panel(std::string path):PyMooseBase(className_, path){}
Panel::Panel(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Panel::Panel(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Panel::Panel(
    const Panel& src,
    std::string name,
    PyMooseBase& parent)
    :PyMooseBase(src, name, parent)
{
}

Panel::Panel(
    const Panel& src,
    std::string name,
    Id& parent)
    :PyMooseBase(src, name, parent)
{
}

Panel::Panel(
    const Panel& src,
    std::string path)
    :PyMooseBase(src, path)
{
}

Panel::Panel(
    const Id& src,
    std::string name,
    Id& parent)
    :PyMooseBase(src, name, parent)
{
}
Panel::Panel(std::string typeName, std::string objectName, Id parentId):
    PyMooseBase(typeName, objectName, parentId)
{
}
   
Panel::Panel(std::string typeName, std::string path):
    PyMooseBase(typeName, path)
{
}

Panel::Panel(std::string typeName, std::string objectName, PyMooseBase& parent):
    PyMooseBase(typeName, objectName, parent)
{
}


Panel::~Panel(){}
const std::string& Panel::getType(){ return className_; }
unsigned int Panel::__get_nPts() const
{
    unsigned int nPts;
    get < unsigned int > (id_(), "nPts",nPts);
    return nPts;
}
unsigned int Panel::__get_nDims() const
{
    unsigned int nDims;
    get < unsigned int > (id_(), "nDims",nDims);
    return nDims;
}
unsigned int Panel::__get_nNeighbors() const
{
    unsigned int nNeighbors;
    get < unsigned int > (id_(), "nNeighbors",nNeighbors);
    return nNeighbors;
}
unsigned int Panel::__get_shapeId() const
{
    unsigned int shapeId;
    get < unsigned int > (id_(), "shapeId",shapeId);
    return shapeId;
}
const vector<double>& Panel::__get_coords() const
{
    vector<double> coords;
    get < vector<double> > (id_(), "coords",coords);
    return coords;
}

#endif
