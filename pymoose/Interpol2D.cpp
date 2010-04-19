#ifndef _pymoose_Interpol2D_cpp
#define _pymoose_Interpol2D_cpp
#include "Interpol2D.h"
using namespace pymoose;
const std::string Interpol2D::className_ = "Interpol2D";
Interpol2D::Interpol2D(Id id):PyMooseBase(id){}
Interpol2D::Interpol2D(std::string path):PyMooseBase(className_, path){}
Interpol2D::Interpol2D(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Interpol2D::Interpol2D(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string path):PyMooseBase(src, path){}
Interpol2D::Interpol2D(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
Interpol2D::Interpol2D(const Id& src, std::string path):PyMooseBase(src, path){}

Interpol2D::~Interpol2D(){}
const std::string& Interpol2D::getType(){ return className_; }
double Interpol2D::__get_ymin() const
{
    double ymin;
    get < double > (id_(), "ymin",ymin);
    return ymin;
}
void Interpol2D::__set_ymin( double ymin )
{
    set < double > (id_(), "ymin", ymin);
}
double Interpol2D::__get_ymax() const
{
    double ymax;
    get < double > (id_(), "ymax",ymax);
    return ymax;
}
void Interpol2D::__set_ymax( double ymax )
{
    set < double > (id_(), "ymax", ymax);
}
int Interpol2D::__get_ydivs() const
{
    int ydivs;
    get < int > (id_(), "ydivs",ydivs);
    return ydivs;
}
void Interpol2D::__set_ydivs( int ydivs )
{
    set < int > (id_(), "ydivs", ydivs);
}
double Interpol2D::__get_dy() const
{
    double dy;
    get < double > (id_(), "dy",dy);
    return dy;
}
void Interpol2D::__set_dy( double dy )
{
    set < double > (id_(), "dy", dy);
}
none Interpol2D::__get_tableVector2D() const
{
    none tableVector2D;
    get < none > (id_(), "tableVector2D",tableVector2D);
    return tableVector2D;
}
void Interpol2D::__set_tableVector2D( none tableVector2D )
{
    set < none > (id_(), "tableVector2D", tableVector2D);
}
#endif
