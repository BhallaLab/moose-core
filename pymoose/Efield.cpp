#ifndef _pymoose_Efield_cpp
#define _pymoose_Efield_cpp
#include "Efield.h"
using namespace pymoose;
const std::string Efield::className_ = "Efield";
Efield::Efield(std::string className, std::string objectName, Id parentId):Neutral(className, objectName, parentId){}
Efield::Efield(std::string className, std::string path):Neutral(className, path){}
Efield::Efield(std::string className, std::string objectName, PyMooseBase& parent):Neutral(className, objectName, parent){}
Efield::Efield(Id id):Neutral(id){}
Efield::Efield(std::string path):Neutral(className_, path){}
Efield::Efield(std::string name, Id parentId):Neutral(className_, name, parentId){}
Efield::Efield(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Efield::Efield(const Efield& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Efield::Efield(const Efield& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Efield::Efield(const Efield& src, std::string path):Neutral(src, path){}
Efield::Efield(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Efield::Efield(const Id& src, std::string path):Neutral(src, path){}
Efield::~Efield(){}
const std::string& Efield::getType(){ return className_; }
double Efield::__get_x() const
{
    double x;
    get < double > (id_(), "x",x);
    return x;
}
void Efield::__set_x( double x )
{
    set < double > (id_(), "x", x);
}
double Efield::__get_y() const
{
    double y;
    get < double > (id_(), "y",y);
    return y;
}
void Efield::__set_y( double y )
{
    set < double > (id_(), "y", y);
}
double Efield::__get_z() const
{
    double z;
    get < double > (id_(), "z",z);
    return z;
}
void Efield::__set_z( double z )
{
    set < double > (id_(), "z", z);
}
double Efield::__get_scale() const
{
    double scale;
    get < double > (id_(), "scale",scale);
    return scale;
}
void Efield::__set_scale( double scale )
{
    set < double > (id_(), "scale", scale);
}
double Efield::__get_potential() const
{
    double potential;
    get < double > (id_(), "potential",potential);
    return potential;
}
#endif
