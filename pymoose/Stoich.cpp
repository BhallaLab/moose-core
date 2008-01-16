#ifndef _pymoose_Stoich_cpp
#define _pymoose_Stoich_cpp

#include "Stoich.h"
using namespace pymoose;
const std::string Stoich::className = "Stoich";
Stoich::Stoich(Id id):PyMooseBase(id){}
Stoich::Stoich(std::string path):PyMooseBase(className, path){}
Stoich::Stoich(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Stoich::Stoich(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Stoich::~Stoich(){}
const std::string& Stoich::getType(){ return className; }
unsigned int Stoich::__get_nMols() const
{
    unsigned int nMols;
    get < unsigned int > (id_(), "nMols",nMols);
    return nMols;
}
void Stoich::__set_nMols( unsigned int nMols )
{
    set < unsigned int > (id_(), "nMols", nMols);
}
unsigned int Stoich::__get_nVarMols() const
{
    unsigned int nVarMols;
    get < unsigned int > (id_(), "nVarMols",nVarMols);
    return nVarMols;
}
void Stoich::__set_nVarMols( unsigned int nVarMols )
{
    set < unsigned int > (id_(), "nVarMols", nVarMols);
}
unsigned int Stoich::__get_nSumTot() const
{
    unsigned int nSumTot;
    get < unsigned int > (id_(), "nSumTot",nSumTot);
    return nSumTot;
}
void Stoich::__set_nSumTot( unsigned int nSumTot )
{
    set < unsigned int > (id_(), "nSumTot", nSumTot);
}
unsigned int Stoich::__get_nBuffered() const
{
    unsigned int nBuffered;
    get < unsigned int > (id_(), "nBuffered",nBuffered);
    return nBuffered;
}
void Stoich::__set_nBuffered( unsigned int nBuffered )
{
    set < unsigned int > (id_(), "nBuffered", nBuffered);
}
unsigned int Stoich::__get_nReacs() const
{
    unsigned int nReacs;
    get < unsigned int > (id_(), "nReacs",nReacs);
    return nReacs;
}
void Stoich::__set_nReacs( unsigned int nReacs )
{
    set < unsigned int > (id_(), "nReacs", nReacs);
}
unsigned int Stoich::__get_nEnz() const
{
    unsigned int nEnz;
    get < unsigned int > (id_(), "nEnz",nEnz);
    return nEnz;
}
void Stoich::__set_nEnz( unsigned int nEnz )
{
    set < unsigned int > (id_(), "nEnz", nEnz);
}
unsigned int Stoich::__get_nMMenz() const
{
    unsigned int nMMenz;
    get < unsigned int > (id_(), "nMMenz",nMMenz);
    return nMMenz;
}
void Stoich::__set_nMMenz( unsigned int nMMenz )
{
    set < unsigned int > (id_(), "nMMenz", nMMenz);
}
unsigned int Stoich::__get_nExternalRates() const
{
    unsigned int nExternalRates;
    get < unsigned int > (id_(), "nExternalRates",nExternalRates);
    return nExternalRates;
}
void Stoich::__set_nExternalRates( unsigned int nExternalRates )
{
    set < unsigned int > (id_(), "nExternalRates", nExternalRates);
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
string Stoich::path() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
string Stoich::path( string path )
{
    set < string > (id_(), "path", path);
    return path;
    
}

unsigned int Stoich::__get_rateVectorSize() const
{
    unsigned int rateVectorSize;
    get < unsigned int > (id_(), "rateVectorSize",rateVectorSize);
    return rateVectorSize;
}
void Stoich::__set_rateVectorSize( unsigned int rateVectorSize )
{
    set < unsigned int > (id_(), "rateVectorSize", rateVectorSize);
}
#endif
