#ifndef _pymoose_Sched1_cpp
#define _pymoose_Sched1_cpp
#include "Sched1.h"
using namespace pymoose;
const std::string Sched1::className = "Sched1";
Sched1::Sched1(Id id):PyMooseBase(id){}
Sched1::Sched1(std::string path):PyMooseBase(className, path){}
Sched1::Sched1(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Sched1::Sched1(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Sched1::~Sched1(){}
const std::string& Sched1::getType(){ return className; }
#endif
