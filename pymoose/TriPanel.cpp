#ifndef _pymoose_TriPanel_cpp
#define _pymoose_TriPanel_cpp
#include "TriPanel.h"
using namespace pymoose;
const std::string TriPanel::className = "TriPanel";
TriPanel::TriPanel(Id id):Panel(id){}
TriPanel::TriPanel(std::string path):Panel(className, path){}
TriPanel::TriPanel(std::string name, Id parentId):Panel(className, name, parentId){}
TriPanel::TriPanel(std::string name, PyMooseBase& parent):Panel(className, name, parent){}
TriPanel::TriPanel(const TriPanel& src, std::string objectName, PyMooseBase& parent):Panel(src, objectName, parent){}
TriPanel::TriPanel(const TriPanel& src, std::string objectName, Id& parent):Panel(src, objectName, parent){}
TriPanel::TriPanel(const TriPanel& src, std::string path):Panel(src, path){}
TriPanel::TriPanel(const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
TriPanel::~TriPanel(){}
const std::string& TriPanel::getType(){ return className; }
#endif
