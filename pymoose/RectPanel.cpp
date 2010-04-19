#ifndef _pymoose_RectPanel_cpp
#define _pymoose_RectPanel_cpp
#include "RectPanel.h"
using namespace pymoose;
const std::string RectPanel::className_ = "RectPanel";
RectPanel::RectPanel(Id id):Panel(id){}
RectPanel::RectPanel(std::string path):Panel(className_, path){}
RectPanel::RectPanel(std::string name, Id parentId):Panel(className_, name, parentId){}
RectPanel::RectPanel(std::string name, PyMooseBase& parent):Panel(className_, name, parent){}
RectPanel::RectPanel(const RectPanel& src, std::string objectName, PyMooseBase& parent):Panel(src, objectName, parent){}
RectPanel::RectPanel(const RectPanel& src, std::string objectName, Id& parent):Panel(src, objectName, parent){}
RectPanel::RectPanel(const RectPanel& src, std::string path):Panel(src, path){}
RectPanel::RectPanel(const Id& src, std::string name, Id& parent):Panel(src, name, parent){}
RectPanel::RectPanel(const Id& src, std::string path):Panel(src, path){}
RectPanel::~RectPanel(){}
const std::string& RectPanel::getType(){ return className_; }
#endif
