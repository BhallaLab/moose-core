#ifndef _pymoose_Reaction_cpp
#define _pymoose_Reaction_cpp
#include "Reaction.h"
using namespace pymoose;
const std::string Reaction::className_ = "Reaction";
Reaction::Reaction(Id id):Neutral(id){}
Reaction::Reaction(std::string path):Neutral(className_, path){}
Reaction::Reaction(std::string name, Id parentId):Neutral(className_, name, parentId){}
Reaction::Reaction(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Reaction::Reaction(const Reaction& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Reaction::Reaction(const Reaction& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Reaction::Reaction(const Reaction& src, std::string path):Neutral(src, path){}
Reaction::Reaction(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Reaction::Reaction(const Id& src, std::string path):Neutral(src, path){}
Reaction::~Reaction(){}
const std::string& Reaction::getType(){ return className_; }
double Reaction::__get_kf() const
{
    double kf;
    get < double > (id_(), "kf",kf);
    return kf;
}
void Reaction::__set_kf( double kf )
{
    set < double > (id_(), "kf", kf);
}
double Reaction::__get_kb() const
{
    double kb;
    get < double > (id_(), "kb",kb);
    return kb;
}
void Reaction::__set_kb( double kb )
{
    set < double > (id_(), "kb", kb);
}
double Reaction::__get_Kf() const
{
    double Kf;
    get < double > (id_(), "Kf",Kf);
    return Kf;
}
void Reaction::__set_Kf( double Kf )
{
    set < double > (id_(), "Kf", Kf);
}
double Reaction::__get_Kb() const
{
    double Kb;
    get < double > (id_(), "Kb",Kb);
    return Kb;
}
void Reaction::__set_Kb( double Kb )
{
    set < double > (id_(), "Kb", Kb);
}
double Reaction::__get_x()
{
    double x;
    get < double > (id_(), "x", x);
    return x;
}

void Reaction::__set_x(double x)
{
    set < double > (id_(), "x", x);
}

double Reaction::__get_y()
{
    double y;
    get < double > (id_(), "y", y);
    return y;
}

void Reaction::__set_y(double y)
{
    set < double > (id_(), "y", y);
}

string Reaction::__get_xtreeTextFg()
{
    string xtreeTextFg;
    get < string > (id_(), "xtree_textfg_req", xtreeTextFg);
    return xtreeTextFg;
}

void Reaction::__set_xtreeTextFg(string xtreeTextFg)
{
    set < string > (id_(), "xtree_textfg_req", xtreeTextFg);
}


#endif
