#ifndef _pymoose_CylPanel_cpp
#define _pymoose_CylPanel_cpp
#include "CylPanel.h"
using namespace pymoose;
const std::string CylPanel::className = "CylPanel";
CylPanel::CylPanel(Id id):Panel(id){}
CylPanel::CylPanel(std::string path):Panel(className, path){}
CylPanel::CylPanel(std::string name, Id parentId):Panel(className, name, parentId){}
CylPanel::CylPanel(std::string name, PyMooseBase& parent):Panel(className, name, parent){}
CylPanel::CylPanel(const CylPanel& src, std::string objectName, PyMooseBase& parent):Panel(src, objectName, parent){}
CylPanel::CylPanel(const CylPanel& src, std::string objectName, Id& parent):Panel(src, objectName, parent){}
CylPanel::CylPanel(const CylPanel& src, std::string path):Panel(src, path){}
CylPanel::CylPanel(const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
CylPanel::~CylPanel(){}
const std::string& CylPanel::getType(){ return className; }
#endif
