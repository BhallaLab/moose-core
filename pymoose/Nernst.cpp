#ifndef _pymoose_Nernst_cpp
#define _pymoose_Nernst_cpp
#include "Nernst.h"
const std::string Nernst::className = "Nernst";
Nernst::Nernst(Id id):PyMooseBase(id){}
Nernst::Nernst(std::string path):PyMooseBase(className, path){}
Nernst::Nernst(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Nernst::Nernst(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Nernst::~Nernst(){}
const std::string& Nernst::getType(){ return className; }
double Nernst::__get_E() const
{
    double E;
    get < double > (Element::element(id_), "E",E);
    return E;
}
void Nernst::__set_E( double E )
{
    set < double > (Element::element(id_), "E", E);
}
double Nernst::__get_Temperature() const
{
    double Temperature;
    get < double > (Element::element(id_), "Temperature",Temperature);
    return Temperature;
}
void Nernst::__set_Temperature( double Temperature )
{
    set < double > (Element::element(id_), "Temperature", Temperature);
}
int Nernst::__get_valence() const
{
    int valence;
    get < int > (Element::element(id_), "valence",valence);
    return valence;
}
void Nernst::__set_valence( int valence )
{
    set < int > (Element::element(id_), "valence", valence);
}
double Nernst::__get_Cin() const
{
    double Cin;
    get < double > (Element::element(id_), "Cin",Cin);
    return Cin;
}
void Nernst::__set_Cin( double Cin )
{
    set < double > (Element::element(id_), "Cin", Cin);
}
double Nernst::__get_Cout() const
{
    double Cout;
    get < double > (Element::element(id_), "Cout",Cout);
    return Cout;
}
void Nernst::__set_Cout( double Cout )
{
    set < double > (Element::element(id_), "Cout", Cout);
}
double Nernst::__get_scale() const
{
    double scale;
    get < double > (Element::element(id_), "scale",scale);
    return scale;
}
void Nernst::__set_scale( double scale )
{
    set < double > (Element::element(id_), "scale", scale);
}
double Nernst::__get_ESrc() const
{
    double ESrc;
    get < double > (Element::element(id_), "ESrc",ESrc);
    return ESrc;
}
void Nernst::__set_ESrc( double ESrc )
{
    set < double > (Element::element(id_), "ESrc", ESrc);
}
double Nernst::__get_CinMsg() const
{
    double CinMsg;
    get < double > (Element::element(id_), "CinMsg",CinMsg);
    return CinMsg;
}
void Nernst::__set_CinMsg( double CinMsg )
{
    set < double > (Element::element(id_), "CinMsg", CinMsg);
}
double Nernst::__get_CoutMsg() const
{
    double CoutMsg;
    get < double > (Element::element(id_), "CoutMsg",CoutMsg);
    return CoutMsg;
}
void Nernst::__set_CoutMsg( double CoutMsg )
{
    set < double > (Element::element(id_), "CoutMsg", CoutMsg);
}
#endif
