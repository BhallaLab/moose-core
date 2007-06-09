#ifndef _pymoose_Molecule_cpp
#define _pymoose_Molecule_cpp
#include "Molecule.h"
const std::string Molecule::className = "Molecule";
Molecule::Molecule(Id id):PyMooseBase(id){}
Molecule::Molecule(std::string path):PyMooseBase(className, path){}
Molecule::Molecule(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Molecule::Molecule(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Molecule::~Molecule(){}
const std::string& Molecule::getType(){ return className; }
double Molecule::__get_nInit() const
{
    double nInit;
    get < double > (Element::element(id_), "nInit",nInit);
    return nInit;
}
void Molecule::__set_nInit( double nInit )
{
    set < double > (Element::element(id_), "nInit", nInit);
}
double Molecule::__get_volumeScale() const
{
    double volumeScale;
    get < double > (Element::element(id_), "volumeScale",volumeScale);
    return volumeScale;
}
void Molecule::__set_volumeScale( double volumeScale )
{
    set < double > (Element::element(id_), "volumeScale", volumeScale);
}
double Molecule::__get_n() const
{
    double n;
    get < double > (Element::element(id_), "n",n);
    return n;
}
void Molecule::__set_n( double n )
{
    set < double > (Element::element(id_), "n", n);
}
int Molecule::__get_mode() const
{
    int mode;
    get < int > (Element::element(id_), "mode",mode);
    return mode;
}
void Molecule::__set_mode( int mode )
{
    set < int > (Element::element(id_), "mode", mode);
}
int Molecule::__get_slave_enable() const
{
    int slave_enable;
    get < int > (Element::element(id_), "slave_enable",slave_enable);
    return slave_enable;
}
void Molecule::__set_slave_enable( int slave_enable )
{
    set < int > (Element::element(id_), "slave_enable", slave_enable);
}
double Molecule::__get_conc() const
{
    double conc;
    get < double > (Element::element(id_), "conc",conc);
    return conc;
}
void Molecule::__set_conc( double conc )
{
    set < double > (Element::element(id_), "conc", conc);
}
double Molecule::__get_concInit() const
{
    double concInit;
    get < double > (Element::element(id_), "concInit",concInit);
    return concInit;
}
void Molecule::__set_concInit( double concInit )
{
    set < double > (Element::element(id_), "concInit", concInit);
}
double Molecule::__get_nSrc() const
{
    double nSrc;
    get < double > (Element::element(id_), "nSrc",nSrc);
    return nSrc;
}
void Molecule::__set_nSrc( double nSrc )
{
    set < double > (Element::element(id_), "nSrc", nSrc);
}
// double,double Molecule::__get_prd() const
// {
//     double,double prd;
//     get < double,double > (Element::element(id_), "prd",prd);
//     return prd;
// }
// void Molecule::__set_prd( double,double prd )
// {
//     set < double,double > (Element::element(id_), "prd", prd);
// }
double Molecule::__get_sumTotal() const
{
    double sumTotal;
    get < double > (Element::element(id_), "sumTotal",sumTotal);
    return sumTotal;
}
void Molecule::__set_sumTotal( double sumTotal )
{
    set < double > (Element::element(id_), "sumTotal", sumTotal);
}
#endif
