#ifndef _pymoose_Stoich_cpp
#define _pymoose_Stoich_cpp

#include "Stoich.h"
using namespace pymoose;
const std::string Stoich::className_ = "Stoich";
Stoich::Stoich(std::string className, std::string objectName, Id parentId):Neutral(className, objectName, parentId){}
Stoich::Stoich(std::string className, std::string path):Neutral(className, path){}
Stoich::Stoich(std::string className, std::string objectName, PyMooseBase& parent): Neutral(className, objectName, parent){} 

Stoich::Stoich(Id id):Neutral(id){}
Stoich::Stoich(std::string path):Neutral(className_, path){}
Stoich::Stoich(std::string name, Id parentId):Neutral(className_, name, parentId){}
Stoich::Stoich(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Stoich::Stoich(const Stoich& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Stoich::Stoich(const Stoich& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Stoich::Stoich(const Stoich& src, std::string path):Neutral(src, path){}
Stoich::Stoich(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Stoich::Stoich(const Id& src, std::string path):Neutral(src, path){}
Stoich::~Stoich(){}
const std::string& Stoich::getType(){ return className_; }
unsigned int Stoich::__get_nMols() const
{
    unsigned int nMols;
    get < unsigned int > (id_(), "nMols",nMols);
    return nMols;
}
unsigned int Stoich::__get_nVarMols() const
{
    unsigned int nVarMols;
    get < unsigned int > (id_(), "nVarMols",nVarMols);
    return nVarMols;
}
unsigned int Stoich::__get_nSumTot() const
{
    unsigned int nSumTot;
    get < unsigned int > (id_(), "nSumTot",nSumTot);
    return nSumTot;
}
unsigned int Stoich::__get_nBuffered() const
{
    unsigned int nBuffered;
    get < unsigned int > (id_(), "nBuffered",nBuffered);
    return nBuffered;
}
unsigned int Stoich::__get_nReacs() const
{
    unsigned int nReacs;
    get < unsigned int > (id_(), "nReacs",nReacs);
    return nReacs;
}
unsigned int Stoich::__get_nEnz() const
{
    unsigned int nEnz;
    get < unsigned int > (id_(), "nEnz",nEnz);
    return nEnz;
}
unsigned int Stoich::__get_nMMenz() const
{
    unsigned int nMMenz;
    get < unsigned int > (id_(), "nMMenz",nMMenz);
    return nMMenz;
}
unsigned int Stoich::__get_nExternalRates() const
{
    unsigned int nExternalRates;
    get < unsigned int > (id_(), "nExternalRates",nExternalRates);
    return nExternalRates;
}
bool Stoich::__get_useOneWayReacs() const
{
    bool useOneWayReacs;
    get < bool > (id_(), "useOneWayReacs",useOneWayReacs);
    return useOneWayReacs;
}
void Stoich::__set_useOneWayReacs( bool useOneWayReacs )
{
    set < bool > (id_(), "useOneWayReacs", useOneWayReacs);
}
// string Stoich::__get_path() const
// {
//     string path;
//     get < string > (id_(), "path",path);
//     return path;
// }
// void Stoich::__set_path( string path )
// {
//     set < string > (id_(), "path", path);
// }
///////////////////////////////
// The following are named so in order to avoid conflict with the
// PyMooseBase path field. 
string Stoich::__get_targetPath() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
void Stoich::__set_targetPath( string path )
{
    set < string > (id_(), "path", path);
}

unsigned int Stoich::__get_rateVectorSize() const
{
    unsigned int rateVectorSize;
    get < unsigned int > (id_(), "rateVectorSize",rateVectorSize);
    return rateVectorSize;
}
#endif
