#ifndef _pymoose_DiskPanel_cpp
#define _pymoose_DiskPanel_cpp
#include "DiskPanel.h"
using namespace pymoose;
const std::string DiskPanel::className_ = "DiskPanel";
DiskPanel::DiskPanel(Id id):Panel(id){}
DiskPanel::DiskPanel(std::string path):Panel(className_, path){}
DiskPanel::DiskPanel(std::string name, Id parentId):Panel(className_, name, parentId){}
DiskPanel::DiskPanel(std::string name, PyMooseBase& parent):Panel(className_, name, parent){}
DiskPanel::DiskPanel(const DiskPanel& src, std::string objectName, PyMooseBase& parent):Panel(src, objectName, parent){}
DiskPanel::DiskPanel(const DiskPanel& src, std::string objectName, Id& parent):Panel(src, objectName, parent){}
DiskPanel::DiskPanel(const DiskPanel& src, std::string path):Panel(src, path){}
DiskPanel::DiskPanel(const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
DiskPanel::DiskPanel(const Id& src, std::string path):Panel(src, path){}
DiskPanel::~DiskPanel(){}
const std::string& DiskPanel::getType(){ return className_; }
#endif
