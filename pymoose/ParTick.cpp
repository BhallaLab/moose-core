#ifndef _pymoose_ParTick_cpp
#define _pymoose_ParTick_cpp
#include "ParTick.h"
using namespace pymoose;
const std::string ParTick::className_ = "ParTick";
ParTick::ParTick(Id id):PyMooseBase(id){}
ParTick::ParTick(std::string path):PyMooseBase(className_, path){}
ParTick::ParTick(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
ParTick::ParTick(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
ParTick::~ParTick(){}
const std::string& ParTick::getType(){ return className_; }
#endif
