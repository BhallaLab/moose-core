#ifndef _pymoose_TickTest_cpp
#define _pymoose_TickTest_cpp
#include "TickTest.h"
const std::string TickTest::className = "TickTest";
TickTest::TickTest(Id id):PyMooseBase(id){}
TickTest::TickTest(std::string path):PyMooseBase(className, path){}
TickTest::TickTest(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
TickTest::TickTest(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
TickTest::~TickTest(){}
const std::string& TickTest::getType(){ return className; }
#endif
