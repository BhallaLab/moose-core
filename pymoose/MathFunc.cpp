#ifndef _pymoose_MathFunc_cpp
#define _pymoose_MathFunc_cpp
#include "MathFunc.h"
using namespace pymoose;
const std::string MathFunc::className = "MathFunc";
MathFunc::MathFunc(Id id):PyMooseBase(id){}
MathFunc::MathFunc(std::string path):PyMooseBase(className, path){}
MathFunc::MathFunc(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
MathFunc::MathFunc(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
MathFunc::~MathFunc(){}
const std::string& MathFunc::getType(){ return className; }
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
double MathFunc::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
void MathFunc::__set_output( double output )
{
    set < double > (id_(), "output", output);
}
double MathFunc::__get_args() const
{
    double args;
    get < double > (id_(), "args",args);
    return args;
}
void MathFunc::__set_args( double args )
{
    set < double > (id_(), "args", args);
}
double MathFunc::__get_arg1() const
{
    double arg1;
    get < double > (id_(), "arg1",arg1);
    return arg1;
}
void MathFunc::__set_arg1( double arg1 )
{
    set < double > (id_(), "arg1", arg1);
}
double MathFunc::__get_arg2() const
{
    double arg2;
    get < double > (id_(), "arg2",arg2);
    return arg2;
}
void MathFunc::__set_arg2( double arg2 )
{
    set < double > (id_(), "arg2", arg2);
}
double MathFunc::__get_arg3() const
{
    double arg3;
    get < double > (id_(), "arg3",arg3);
    return arg3;
}
void MathFunc::__set_arg3( double arg3 )
{
    set < double > (id_(), "arg3", arg3);
}
double MathFunc::__get_arg4() const
{
    double arg4;
    get < double > (id_(), "arg4",arg4);
    return arg4;
}
void MathFunc::__set_arg4( double arg4 )
{
    set < double > (id_(), "arg4", arg4);
}
#endif
