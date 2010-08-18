#ifndef _pymoose_MathFunc_cpp
#define _pymoose_MathFunc_cpp
#include "MathFunc.h"
using namespace pymoose;
const std::string MathFunc::className_ = "MathFunc";
MathFunc::MathFunc(Id id):Neutral(id){}
MathFunc::MathFunc(std::string path):Neutral(className_, path){}
MathFunc::MathFunc(std::string name, Id parentId):Neutral(className_, name, parentId){}
MathFunc::MathFunc(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
MathFunc::MathFunc(const MathFunc& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
MathFunc::MathFunc(const MathFunc& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
MathFunc::MathFunc(const MathFunc& src, std::string path):Neutral(src, path){}
MathFunc::MathFunc(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
MathFunc::MathFunc(const Id& src, std::string path):Neutral(src, path){}
MathFunc::~MathFunc(){}
const std::string& MathFunc::getType(){ return className_; }
string MathFunc::__get_mathML() const
{
    string mathML;
    get < string > (id_(), "mathML",mathML);
    return mathML;
}
void MathFunc::__set_mathML( string mathML )
{
    set < string > (id_(), "mathML", mathML);
}
string MathFunc::__get_function() const
{
    string function;
    get < string > (id_(), "function",function);
    return function;
}
void MathFunc::__set_function( string function )
{
    set < string > (id_(), "function", function);
}
double MathFunc::__get_result() const
{
    double result;
    get < double > (id_(), "result",result);
    return result;
}
void MathFunc::__set_result( double result )
{
    set < double > (id_(), "result", result);
}
#endif
