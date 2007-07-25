#ifndef _pymoose_HSolve_cpp
#define _pymoose_HSolve_cpp
#include "HSolve.h"
const std::string HSolve::className = "HSolve";
HSolve::HSolve(Id id):PyMooseBase(id){}
HSolve::HSolve(std::string path):PyMooseBase(className, path){}
HSolve::HSolve(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HSolve::HSolve(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
HSolve::~HSolve(){}
const std::string& HSolve::getType(){ return className; }

const string HSolve::__get_seed_path() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
void HSolve::__set_seed_path( const string path ) const
{
    set < string > (id_(), "path", path);
}
int HSolve::__get_NDiv() const
{
    int NDiv;
    get < int > (id_(), "NDiv",NDiv);
    return NDiv;
}
void HSolve::__set_NDiv( int NDiv )
{
    set < int > (id_(), "NDiv", NDiv);
}
double HSolve::__get_VLo() const
{
    double VLo;
    get < double > (id_(), "VLo",VLo);
    return VLo;
}
void HSolve::__set_VLo( double VLo )
{
    set < double > (id_(), "VLo", VLo);
}
double HSolve::__get_VHi() const
{
    double VHi;
    get < double > (id_(), "VHi",VHi);
    return VHi;
}
void HSolve::__set_VHi( double VHi )
{
    set < double > (id_(), "VHi", VHi);
}

#endif
