#ifndef _pymoose_DifShell_cpp
#define _pymoose_DifShell_cpp
#include "DifShell.h"
using namespace pymoose;
const std::string DifShell::className_ = "DifShell";
DifShell::DifShell(Id id):Neutral(id){}
DifShell::DifShell(std::string path):Neutral(className_, path){}
DifShell::DifShell(std::string name, Id parentId):Neutral(className_, name, parentId){}
DifShell::DifShell(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
DifShell::DifShell(const DifShell& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
DifShell::DifShell(const DifShell& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
DifShell::DifShell(const DifShell& src, std::string path):Neutral(src, path){}
DifShell::DifShell(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
DifShell::DifShell(const Id& src, std::string path):Neutral(src, path){}
DifShell::~DifShell(){}
const std::string& DifShell::getType(){ return className_; }
double DifShell::__get_C() const
{
    double C;
    get < double > (id_(), "C",C);
    return C;
}
double DifShell::__get_Ceq() const
{
    double Ceq;
    get < double > (id_(), "Ceq",Ceq);
    return Ceq;
}
void DifShell::__set_Ceq( double Ceq )
{
    set < double > (id_(), "Ceq", Ceq);
}
double DifShell::__get_D() const
{
    double D;
    get < double > (id_(), "D",D);
    return D;
}
void DifShell::__set_D( double D )
{
    set < double > (id_(), "D", D);
}
double DifShell::__get_valence() const
{
    double valence;
    get < double > (id_(), "valence",valence);
    return valence;
}
void DifShell::__set_valence( double valence )
{
    set < double > (id_(), "valence", valence);
}
double DifShell::__get_leak() const
{
    double leak;
    get < double > (id_(), "leak",leak);
    return leak;
}
void DifShell::__set_leak( double leak )
{
    set < double > (id_(), "leak", leak);
}
unsigned int DifShell::__get_shapeMode() const
{
    unsigned int shapeMode;
    get < unsigned int > (id_(), "shapeMode",shapeMode);
    return shapeMode;
}
void DifShell::__set_shapeMode( unsigned int shapeMode )
{
    set < unsigned int > (id_(), "shapeMode", shapeMode);
}
double DifShell::__get_length() const
{
    double length;
    get < double > (id_(), "length",length);
    return length;
}
void DifShell::__set_length( double length )
{
    set < double > (id_(), "length", length);
}
double DifShell::__get_diameter() const
{
    double diameter;
    get < double > (id_(), "diameter",diameter);
    return diameter;
}
void DifShell::__set_diameter( double diameter )
{
    set < double > (id_(), "diameter", diameter);
}
double DifShell::__get_thickness() const
{
    double thickness;
    get < double > (id_(), "thickness",thickness);
    return thickness;
}
void DifShell::__set_thickness( double thickness )
{
    set < double > (id_(), "thickness", thickness);
}
double DifShell::__get_volume() const
{
    double volume;
    get < double > (id_(), "volume",volume);
    return volume;
}
void DifShell::__set_volume( double volume )
{
    set < double > (id_(), "volume", volume);
}
double DifShell::__get_outerArea() const
{
    double outerArea;
    get < double > (id_(), "outerArea",outerArea);
    return outerArea;
}
void DifShell::__set_outerArea( double outerArea )
{
    set < double > (id_(), "outerArea", outerArea);
}
double DifShell::__get_innerArea() const
{
    double innerArea;
    get < double > (id_(), "innerArea",innerArea);
    return innerArea;
}
void DifShell::__set_innerArea( double innerArea )
{
    set < double > (id_(), "innerArea", innerArea);
}
#endif
