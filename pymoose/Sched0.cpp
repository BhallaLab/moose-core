#ifndef _pymoose_Sched0_cpp
#define _pymoose_Sched0_cpp
#include "Sched0.h"
const std::string Sched0::className = "Sched0";
Sched0::Sched0(Id id):PyMooseBase(id){}
Sched0::Sched0(std::string path):PyMooseBase(className, path){}
Sched0::Sched0(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Sched0::Sched0(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Sched0::~Sched0(){}
const std::string& Sched0::getType(){ return className; }
#endif
