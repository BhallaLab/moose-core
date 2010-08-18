#ifndef _pymoose_Panel_cpp
#define _pymoose_Panel_cpp
#include "Panel.h"
using namespace pymoose;
const std::string Panel::className_ = "Panel";
Panel::Panel(std::string typeName, std::string objectName, Id parentId): Neutral(typeName, objectName, parentId){}
Panel::Panel(std::string typeName, std::string path): Neutral(typeName, path){}
Panel::Panel(std::string typeName, std::string objectName, PyMooseBase& parent): Neutral(typeName, objectName, parent){}

Panel::Panel(Id id):Neutral(id){}
Panel::Panel(std::string path):Neutral(className_, path){}
Panel::Panel(std::string name, Id parentId):Neutral(className_, name, parentId){}
Panel::Panel(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Panel::Panel(const Panel& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Panel::Panel(const Panel& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Panel::Panel(const Panel& src, std::string path):Neutral(src, path){}
Panel::Panel(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Panel::Panel(const Id& src, std::string path):Neutral(src, path){}
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
vector<double> Panel::__get_coords() const
{
    vector<double> coords;
    get < vector<double> > (id_(), "coords",coords);
    return coords;
}

#endif
