#ifndef _pymoose_TickTest_cpp
#define _pymoose_TickTest_cpp
#include "TickTest.h"
using namespace pymoose;

const std::string TickTest::className_ = "TickTest";
TickTest::TickTest(Id id):PyMooseBase(id){}
TickTest::TickTest(std::string path):PyMooseBase(className_, path){}
TickTest::TickTest(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
TickTest::TickTest(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
TickTest::~TickTest(){}
const std::string& TickTest::getType(){ return className_; }
#endif
