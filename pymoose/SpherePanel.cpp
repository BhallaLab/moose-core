#ifndef _pymoose_SpherePanel_cpp
#define _pymoose_SpherePanel_cpp
#include "SpherePanel.h"
using namespace pymoose;
const std::string SpherePanel::className = "SpherePanel";
SpherePanel::SpherePanel(Id id):Panel(id){}
SpherePanel::SpherePanel(std::string path):Panel(className, path){}
SpherePanel::SpherePanel(std::string name, Id parentId):Panel(className, name, parentId){}
SpherePanel::SpherePanel(std::string name, PyMooseBase& parent):Panel(className, name, parent){}
SpherePanel::SpherePanel(const SpherePanel& src, std::string objectName, PyMooseBase& parent):Panel(src, objectName, parent){}
SpherePanel::SpherePanel(const SpherePanel& src, std::string objectName, Id& parent):Panel(src, objectName, parent){}
SpherePanel::SpherePanel(const SpherePanel& src, std::string path):Panel(src, path){}
SpherePanel::SpherePanel(const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
SpherePanel::~SpherePanel(){}
const std::string& SpherePanel::getType(){ return className; }
#endif
