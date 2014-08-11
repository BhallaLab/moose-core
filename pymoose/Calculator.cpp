#ifndef _pymoose_Calculator_cpp
#define _pymoose_Calculator_cpp
#include "Calculator.h"
using namespace pymoose;
const std::string Calculator::className_ = "Calculator";
Calculator::Calculator(std::string className, std::string objectName, Id parentId):Neutral(className, objectName, parentId){}
Calculator::Calculator(std::string className, std::string path):Neutral(className, path){}
Calculator::Calculator(std::string className, std::string objectName, PyMooseBase& parent):Neutral(className, objectName, parent){}
Calculator::Calculator(Id id):Neutral(id){}
Calculator::Calculator(std::string path):Neutral(className_, path){}
Calculator::Calculator(std::string name, Id parentId):Neutral(className_, name, parentId){}
Calculator::Calculator(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Calculator::Calculator(const Calculator& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Calculator::Calculator(const Calculator& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Calculator::Calculator(const Calculator& src, std::string path):Neutral(src, path){}
Calculator::Calculator(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Calculator::Calculator(const Id& src, std::string path):Neutral(src, path){}
Calculator::~Calculator(){}
const std::string& Calculator::getType(){ return className_; }
double Calculator::__get_initValue() const
{
    double initValue;
    get < double > (id_(), "initValue",initValue);
    return initValue;
}
void Calculator::__set_initValue( double initValue )
{
    set < double > (id_(), "initValue", initValue);
}
double Calculator::__get_value() const
{
    double value;
    get < double > (id_(), "value",value);
    return value;
}
#endif
