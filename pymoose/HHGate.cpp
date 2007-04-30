#ifndef _pymoose_HHGate_cpp
#define _pymoose_HHGate_cpp
#include "HHGate.h"
const std::string HHGate::className = "HHGate";
HHGate::HHGate(Id id):PyMooseBase(id){}
HHGate::HHGate(std::string path):PyMooseBase(className, path){}
HHGate::HHGate(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HHGate::HHGate(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
HHGate::~HHGate(){}
const std::string& HHGate::getType(){ return className; }

// Manually edited part
Id HHGate::getA() const
{
    return PyMooseBase::pathToId(this->path()+"/A");    
}
Id HHGate::getB() const
{
    return PyMooseBase::pathToId(this->path()+"/B");
}

#endif
