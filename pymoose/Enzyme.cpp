#ifndef _pymoose_Enzyme_cpp
#define _pymoose_Enzyme_cpp
#include "Enzyme.h"
using namespace pymoose;
const std::string Enzyme::className_ = "Enzyme";
Enzyme::Enzyme(Id id):Neutral(id){}
Enzyme::Enzyme(std::string path):Neutral(className_, path){}
Enzyme::Enzyme(std::string name, Id parentId):Neutral(className_, name, parentId){}
Enzyme::Enzyme(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Enzyme::Enzyme(const Enzyme& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Enzyme::Enzyme(const Enzyme& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Enzyme::Enzyme(const Enzyme& src, std::string path):Neutral(src, path){}
Enzyme::Enzyme(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Enzyme::Enzyme(const Id& src, std::string path):Neutral(src, path){}
Enzyme::~Enzyme(){}
const std::string& Enzyme::getType(){ return className_; }
double Enzyme::__get_k1() const
{
    double k1;
    get < double > (id_(), "k1",k1);
    return k1;
}
void Enzyme::__set_k1( double k1 )
{
    set < double > (id_(), "k1", k1);
}
double Enzyme::__get_k2() const
{
    double k2;
    get < double > (id_(), "k2",k2);
    return k2;
}
void Enzyme::__set_k2( double k2 )
{
    set < double > (id_(), "k2", k2);
}
double Enzyme::__get_k3() const
{
    double k3;
    get < double > (id_(), "k3",k3);
    return k3;
}
void Enzyme::__set_k3( double k3 )
{
    set < double > (id_(), "k3", k3);
}
double Enzyme::__get_Km() const
{
    double Km;
    get < double > (id_(), "Km",Km);
    return Km;
}
void Enzyme::__set_Km( double Km )
{
    set < double > (id_(), "Km", Km);
}
double Enzyme::__get_kcat() const
{
    double kcat;
    get < double > (id_(), "kcat",kcat);
    return kcat;
}
void Enzyme::__set_kcat( double kcat )
{
    set < double > (id_(), "kcat", kcat);
}
bool Enzyme::__get_mode() const
{
    bool mode;
    get < bool > (id_(), "mode",mode);
    return mode;
}
void Enzyme::__set_mode( bool mode )
{
    set < bool > (id_(), "mode", mode);
}
// double,double Enzyme::__get_prd() const
// {
//     double,double prd;
//     get < double,double > (id_(), "prd",prd);
//     return prd;
// }
// void Enzyme::__set_prd( double,double prd )
// {
//     set < double,double > (id_(), "prd", prd);
// }
double Enzyme::__get_scaleKm() const
{
    double scaleKm;
    get < double > (id_(), "scaleKm",scaleKm);
    return scaleKm;
}
void Enzyme::__set_scaleKm( double scaleKm )
{
    set < double > (id_(), "scaleKm", scaleKm);
}
double Enzyme::__get_scaleKcat() const
{
    double scaleKcat;
    get < double > (id_(), "scaleKcat",scaleKcat);
    return scaleKcat;
}
void Enzyme::__set_scaleKcat( double scaleKcat )
{
    set < double > (id_(), "scaleKcat", scaleKcat);
}
double Enzyme::__get_intramol() const
{
    double intramol;
    get < double > (id_(), "intramol",intramol);
    return intramol;
}
void Enzyme::__set_intramol( double intramol )
{
    set < double > (id_(), "intramol", intramol);
}

double Enzyme::__get_x()
{
    double x;
    get < double > (id_(), "x", x);
    return x;
}

void Enzyme::__set_x(double x)
{
    set < double > (id_(), "x", x);
}

double Enzyme::__get_y()
{
    double y;
    get < double > (id_(), "y", y);
    return y;
}

void Enzyme::__set_y(double y)
{
    set < double > (id_(), "y", y);
}

string Enzyme::__get_xtreeTextFg()
{
    string xtreeTextFg;
    get < string > (id_(), "xtree_textfg_req", xtreeTextFg);
    return xtreeTextFg;
}

void Enzyme::__set_xtreeTextFg(string xtreeTextFg)
{
    set < string > (id_(), "xtree_textfg_req", xtreeTextFg);
}

#endif
