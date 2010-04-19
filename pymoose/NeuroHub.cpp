#ifndef _pymoose_NeuroHub_cpp
#define _pymoose_NeuroHub_cpp
#include "NeuroHub.h"
using namespace pymoose;
const std::string NeuroHub::className_ = "NeuroHub";
NeuroHub::NeuroHub(Id id):PyMooseBase(id){}
NeuroHub::NeuroHub(std::string path):PyMooseBase(className_, path){}
NeuroHub::NeuroHub(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
NeuroHub::NeuroHub(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
NeuroHub::NeuroHub(const NeuroHub& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
NeuroHub::NeuroHub(const NeuroHub& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
NeuroHub::NeuroHub(const NeuroHub& src, std::string path):PyMooseBase(src, path){}
NeuroHub::NeuroHub(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
NeuroHub::NeuroHub(const Id& src, std::string path):PyMooseBase(src, path){}

NeuroHub::~NeuroHub(){}
const std::string& NeuroHub::getType(){ return className_; }
#endif
