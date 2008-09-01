#ifndef _pymoose_DiskPanel_cpp
#define _pymoose_DiskPanel_cpp
#include "DiskPanel.h"
using namespace pymoose;
const std::string DiskPanel::className = "DiskPanel";
DiskPanel::DiskPanel(Id id):Panel(id){}
DiskPanel::DiskPanel(std::string path):Panel(className, path){}
DiskPanel::DiskPanel(std::string name, Id parentId):Panel(className, name, parentId){}
DiskPanel::DiskPanel(std::string name, PyMooseBase& parent):Panel(className, name, parent){}
DiskPanel::DiskPanel( const DiskPanel& src, std::string name, PyMooseBase& parent):Panel(src, name, parent){}
DiskPanel::DiskPanel( const DiskPanel& src, std::string name, Id& parent):Panel(src, name, parent){}
DiskPanel::DiskPanel( const DiskPanel& src, std::string path):Panel(src, path){}
DiskPanel::DiskPanel( const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
DiskPanel::~DiskPanel(){}
const std::string& DiskPanel::getType(){ return className; }
#endif
