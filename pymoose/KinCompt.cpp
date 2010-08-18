#ifndef _pymoose_KinCompt_cpp
#define _pymoose_KinCompt_cpp
#include "KinCompt.h"
using namespace pymoose;
const std::string KinCompt::className_ = "KinCompt";

KinCompt::KinCompt(std::string className, std::string objectName, Id parentId):Neutral(className, objectName, parentId){}
KinCompt::KinCompt(std::string className, std::string path):Neutral(className, path){}
KinCompt::KinCompt(std::string className, std::string objectName, PyMooseBase& parent): Neutral(className, objectName, parent){}

KinCompt::KinCompt(Id id):Neutral(id){}
KinCompt::KinCompt(std::string path):Neutral(className_, path){}
KinCompt::KinCompt(std::string name, Id parentId):Neutral(className_, name, parentId){}
KinCompt::KinCompt(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
KinCompt::KinCompt(const KinCompt& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
KinCompt::KinCompt(const KinCompt& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
KinCompt::KinCompt(const KinCompt& src, std::string path):Neutral(src, path){}
KinCompt::KinCompt(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
KinCompt::KinCompt(const Id& src, std::string path):Neutral(src, path){}
KinCompt::~KinCompt(){}
const std::string& KinCompt::getType(){ return className_; }
double KinCompt::__get_volume() const
{
    double volume;
    get < double > (id_(), "volume",volume);
    return volume;
}
void KinCompt::__set_volume( double volume )
{
    set < double > (id_(), "volume", volume);
}
double KinCompt::__get_area() const
{
    double area;
    get < double > (id_(), "area",area);
    return area;
}
void KinCompt::__set_area( double area )
{
    set < double > (id_(), "area", area);
}
double KinCompt::__get_perimeter() const
{
    double perimeter;
    get < double > (id_(), "perimeter",perimeter);
    return perimeter;
}
void KinCompt::__set_perimeter( double perimeter )
{
    set < double > (id_(), "perimeter", perimeter);
}
double KinCompt::__get_size() const
{
    double size;
    get < double > (id_(), "size",size);
    return size;
}
void KinCompt::__set_size( double size )
{
    set < double > (id_(), "size", size);
}
unsigned int KinCompt::__get_numDimensions() const
{
    unsigned int numDimensions;
    get < unsigned int > (id_(), "numDimensions",numDimensions);
    return numDimensions;
}
void KinCompt::__set_numDimensions( unsigned int numDimensions )
{
    set < unsigned int > (id_(), "numDimensions", numDimensions);
}

double KinCompt::__get_x()
{
    double x;
    get < double > (id_(), "x", x);
    return x;
}

void KinCompt::__set_x(double x)
{
    set < double > (id_(), "x", x);
}

double KinCompt::__get_y()
{
    double y;
    get < double > (id_(), "y", y);
    return y;
}

void KinCompt::__set_y(double y)
{
    set < double > (id_(), "y", y);
}


#endif
