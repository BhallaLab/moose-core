#ifndef _pymoose_ParTick_cpp
#define _pymoose_ParTick_cpp
#include "ParTick.h"
using namespace pymoose;
const std::string ParTick::className = "ParTick";
ParTick::ParTick(Id id):PyMooseBase(id){}
ParTick::ParTick(std::string path):PyMooseBase(className, path){}
ParTick::ParTick(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
ParTick::ParTick(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
ParTick::~ParTick(){}
const std::string& ParTick::getType(){ return className; }
#endif
