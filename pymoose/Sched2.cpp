#ifndef _pymoose_Sched2_cpp
#define _pymoose_Sched2_cpp
#include "Sched2.h"
using namespace pymoose;
const std::string Sched2::className = "Sched2";
Sched2::Sched2(Id id):PyMooseBase(id){}
Sched2::Sched2(std::string path):PyMooseBase(className, path){}
Sched2::Sched2(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Sched2::Sched2(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Sched2::~Sched2(){}
const std::string& Sched2::getType(){ return className; }
#endif
