#ifndef _pymoose_Leakage_cpp
#define _pymoose_Leakage_cpp
#include "Leakage.h"
using namespace pymoose;
const std::string Leakage::className_ = "Leakage";
Leakage::Leakage(Id id):Neutral(id){}
Leakage::Leakage(std::string path):Neutral(className_, path){}
Leakage::Leakage(std::string name, Id parentId):Neutral(className_, name, parentId){}
Leakage::Leakage(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Leakage::Leakage(const Leakage& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Leakage::Leakage(const Leakage& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Leakage::Leakage(const Leakage& src, std::string path):Neutral(src, path){}
Leakage::Leakage(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Leakage::Leakage(const Id& src, std::string path):Neutral(src, path){}
Leakage::~Leakage(){}
const std::string& Leakage::getType(){ return className_; }
double Leakage::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void Leakage::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double Leakage::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void Leakage::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double Leakage::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
double Leakage::__get_activation() const
{
    double activation;
    get < double > (id_(), "activation",activation);
    return activation;
}
void Leakage::__set_activation( double activation )
{
    set < double > (id_(), "activation", activation);
}
#endif
