#ifndef _pymoose_HHGate2D_cpp
#define _pymoose_HHGate2D_cpp
#include "HHGate2D.h"
using namespace pymoose;
const std::string HHGate2D::className_ = "HHGate2D";
HHGate2D::HHGate2D(Id id):PyMooseBase(id){}
HHGate2D::HHGate2D(std::string path):PyMooseBase(className_, path){}
HHGate2D::HHGate2D(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
HHGate2D::HHGate2D(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
HHGate2D::HHGate2D(const HHGate2D& src, std::string path):PyMooseBase(src, path){}
HHGate2D::HHGate2D(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
HHGate2D::~HHGate2D(){}
const std::string& HHGate2D::getType(){ return className_; }
#endif
