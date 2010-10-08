#ifndef _pymoose_Molecule_cpp
#define _pymoose_Molecule_cpp
#include "Molecule.h"
using namespace pymoose;
const std::string Molecule::className_ = "Molecule";
Molecule::Molecule(Id id):Neutral(id){}
Molecule::Molecule(std::string path):Neutral(className_, path){}
Molecule::Molecule(std::string name, Id parentId):Neutral(className_, name, parentId){}
Molecule::Molecule(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Molecule::Molecule(const Molecule& src, std::string objectName,  PyMooseBase& parent):Neutral(src, objectName, parent){}

Molecule::Molecule(const Molecule& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Molecule::Molecule(const Molecule& src, std::string path):Neutral(src, path)
{
}

Molecule::Molecule(const Id& src, string name, Id& parent):Neutral(src, name, parent)
{
}
Molecule::Molecule(const Id& src, string path):Neutral(src, path)
{
}
Molecule::~Molecule(){}
const std::string& Molecule::getType(){ return className_; }
double Molecule::__get_nInit() const
{
    double nInit;
    get < double > (id_(), "nInit",nInit);
    return nInit;
}
void Molecule::__set_nInit( double nInit )
{
    set < double > (id_(), "nInit", nInit);
}
double Molecule::__get_volumeScale() const
{
    double volumeScale;
    get < double > (id_(), "volumeScale",volumeScale);
    return volumeScale;
}
void Molecule::__set_volumeScale( double volumeScale )
{
    set < double > (id_(), "volumeScale", volumeScale);
}
double Molecule::__get_n() const
{
    double n;
    get < double > (id_(), "n",n);
    return n;
}
void Molecule::__set_n( double n )
{
    set < double > (id_(), "n", n);
}
int Molecule::__get_mode() const
{
    int mode;
    get < int > (id_(), "mode",mode);
    return mode;
}
void Molecule::__set_mode( int mode )
{
    set < int > (id_(), "mode", mode);
}
int Molecule::__get_slave_enable() const
{
    int slave_enable;
    get < int > (id_(), "slave_enable",slave_enable);
    return slave_enable;
}
void Molecule::__set_slave_enable( int slave_enable )
{
    set < int > (id_(), "slave_enable", slave_enable);
}
double Molecule::__get_conc() const
{
    double conc;
    get < double > (id_(), "conc",conc);
    return conc;
}
void Molecule::__set_conc( double conc )
{
    set < double > (id_(), "conc", conc);
}
double Molecule::__get_concInit() const
{
    double concInit;
    get < double > (id_(), "concInit",concInit);
    return concInit;
}
void Molecule::__set_concInit( double concInit )
{
    set < double > (id_(), "concInit", concInit);
}
double Molecule::__get_nSrc() const
{
    double nSrc;
    get < double > (id_(), "nSrc",nSrc);
    return nSrc;
}
void Molecule::__set_nSrc( double nSrc )
{
    set < double > (id_(), "nSrc", nSrc);
}
// double,double Molecule::__get_prd() const
// {
//     double,double prd;
//     get < double,double > (id_(), "prd",prd);
//     return prd;
// }
// void Molecule::__set_prd( double,double prd )
// {
//     set < double,double > (id_(), "prd", prd);
// }
double Molecule::__get_sumTotal() const
{
    double sumTotal;
    get < double > (id_(), "sumTotal",sumTotal);
    return sumTotal;
}
void Molecule::__set_sumTotal( double sumTotal )
{
    set < double > (id_(), "sumTotal", sumTotal);
}

double Molecule::__get_x()
{
    double x;
    get < double > (id_(), "x", x);
    return x;
}

void Molecule::__set_x(double x)
{
    set < double > (id_(), "x", x);
}

double Molecule::__get_y()
{
    double y;
    get < double > (id_(), "y", y);
    return y;
}

void Molecule::__set_y(double y)
{
    set < double > (id_(), "y", y);
}
double Molecule::__get_D()
{
    double y;
    get < double > (id_(), "D", y);
    return y;
}

void Molecule::__set_D(double y)
{
    set < double > (id_(), "D", y);
}

string Molecule::__get_xtreeTextFg()
{
    string xtreeTextFg;
    get < string > (id_(), "xtree_textfg_req", xtreeTextFg);
    return xtreeTextFg;
}

void Molecule::__set_xtreeTextFg(string xtreeTextFg)
{
    set < string > (id_(), "xtree_textfg_req", xtreeTextFg);
}

#endif
