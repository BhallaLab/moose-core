#ifndef _pymoose_Geometry_cpp
#define _pymoose_Geometry_cpp
#include "Geometry.h"
using namespace pymoose;
const std::string Geometry::className_ = "Geometry";
Geometry::Geometry(Id id):PyMooseBase(id){}
Geometry::Geometry(std::string path):PyMooseBase(className_, path){}
Geometry::Geometry(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Geometry::Geometry(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Geometry::Geometry( const Geometry& src, std::string name, PyMooseBase& parent):PyMooseBase(src, name, parent)
{
}    
Geometry::Geometry( const Geometry& src, std::string name, Id& parent):PyMooseBase(src, name, parent)
{
}

Geometry::Geometry( const Geometry& src, std::string path):PyMooseBase(src, path)
{
}

Geometry::Geometry( const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent)
{
}
Geometry::Geometry( const Id& src, std::string path):PyMooseBase(src, path)
{
}


Geometry::~Geometry(){}
const std::string& Geometry::getType(){ return className_; }
double Geometry::__get_epsilon() const
{
    double epsilon;
    get < double > (id_(), "epsilon",epsilon);
    return epsilon;
}
void Geometry::__set_epsilon( double epsilon )
{
    set < double > (id_(), "epsilon", epsilon);
}
double Geometry::__get_neighdist() const
{
    double neighdist;
    get < double > (id_(), "neighdist",neighdist);
    return neighdist;
}
void Geometry::__set_neighdist( double neighdist )
{
    set < double > (id_(), "neighdist", neighdist);
}
#endif
