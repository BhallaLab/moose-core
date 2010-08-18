#ifndef _pymoose_Mg_block_cpp
#define _pymoose_Mg_block_cpp
#include "Mg_block.h"
using namespace pymoose;
const std::string Mg_block::className_ = "Mg_block";
Mg_block::Mg_block(Id id):Neutral(id){}
Mg_block::Mg_block(std::string path):Neutral(className_, path){}
Mg_block::Mg_block(std::string name, Id parentId):Neutral(className_, name, parentId){}
Mg_block::Mg_block(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Mg_block::Mg_block(const Mg_block& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Mg_block::Mg_block(const Mg_block& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Mg_block::Mg_block(const Mg_block& src, std::string path):Neutral(src, path){}
Mg_block::Mg_block(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Mg_block::Mg_block(const Id& src, std::string path):Neutral(src, path){}
Mg_block::~Mg_block(){}
const std::string& Mg_block::getType(){ return className_; }
double Mg_block::__get_KMg_A() const
{
    double KMg_A;
    get < double > (id_(), "KMg_A",KMg_A);
    return KMg_A;
}
void Mg_block::__set_KMg_A( double KMg_A )
{
    set < double > (id_(), "KMg_A", KMg_A);
}
double Mg_block::__get_KMg_B() const
{
    double KMg_B;
    get < double > (id_(), "KMg_B",KMg_B);
    return KMg_B;
}
void Mg_block::__set_KMg_B( double KMg_B )
{
    set < double > (id_(), "KMg_B", KMg_B);
}
double Mg_block::__get_CMg() const
{
    double CMg;
    get < double > (id_(), "CMg",CMg);
    return CMg;
}
void Mg_block::__set_CMg( double CMg )
{
    set < double > (id_(), "CMg", CMg);
}
double Mg_block::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
void Mg_block::__set_Ik( double Ik )
{
    set < double > (id_(), "Ik", Ik);
}
double Mg_block::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void Mg_block::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double Mg_block::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void Mg_block::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double Mg_block::__get_Zk() const
{
    double Zk;
    get < double > (id_(), "Zk",Zk);
    return Zk;
}
void Mg_block::__set_Zk( double Zk )
{
    set < double > (id_(), "Zk", Zk);
}
#endif
