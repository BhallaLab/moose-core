#ifndef _pymoose_HHGate2D_cpp
#define _pymoose_HHGate2D_cpp
#include "HHGate2D.h"
using namespace pymoose;
const std::string HHGate2D::className_ = "HHGate2D";
HHGate2D::HHGate2D(Id id):HHGate(id){}
HHGate2D::HHGate2D(std::string path):HHGate(className_, path){}
HHGate2D::HHGate2D(std::string name, Id parentId):HHGate(className_, name, parentId){}
HHGate2D::HHGate2D(std::string name, PyMooseBase& parent):HHGate(className_, name, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string objectName, PyMooseBase& parent):HHGate(src, objectName, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string objectName, Id& parent):HHGate(src, objectName, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string path):HHGate(src, path){}
HHGate2D::HHGate2D(const Id& src, std::string name, Id& parent):HHGate(src, name, parent){}
HHGate2D::HHGate2D(const Id& src, std::string path):HHGate(src, path){}
HHGate2D::~HHGate2D(){}
const std::string& HHGate2D::getType(){ return className_; }
// Manually edited part
Interpol2D* HHGate2D::__get_A() const
{
    return new Interpol2D(PyMooseBase::pathToId(this->__get_path()+"/A"));    
}
Interpol2D* HHGate2D::__get_B() const
{
    return new Interpol2D(PyMooseBase::pathToId(this->__get_path()+"/B"));
}
// till here
#endif
