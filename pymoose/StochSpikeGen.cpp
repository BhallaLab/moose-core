#ifndef _pymoose_StochSpikeGen_cpp
#define _pymoose_StochSpikeGen_cpp
#include "StochSpikeGen.h"
using namespace pymoose;
const std::string StochSpikeGen::className_ = "StochSpikeGen";
StochSpikeGen::StochSpikeGen(std::string className, std::string objectName, Id parentId):SpikeGen(className, objectName, parentId){}
StochSpikeGen::StochSpikeGen(std::string className, std::string path):SpikeGen(className, path){}
StochSpikeGen::StochSpikeGen(std::string className, std::string objectName, PyMooseBase& parent):SpikeGen(className, objectName, parent){}
StochSpikeGen::StochSpikeGen(Id id):SpikeGen(id){}
StochSpikeGen::StochSpikeGen(std::string path):SpikeGen(className_, path){}
StochSpikeGen::StochSpikeGen(std::string name, Id parentId):SpikeGen(className_, name, parentId){}
StochSpikeGen::StochSpikeGen(std::string name, PyMooseBase& parent):SpikeGen(className_, name, parent){}
StochSpikeGen::StochSpikeGen(const StochSpikeGen& src, std::string objectName, PyMooseBase& parent):SpikeGen(src, objectName, parent){}
StochSpikeGen::StochSpikeGen(const StochSpikeGen& src, std::string objectName, Id& parent):SpikeGen(src, objectName, parent){}
StochSpikeGen::StochSpikeGen(const StochSpikeGen& src, std::string path):SpikeGen(src, path){}
StochSpikeGen::StochSpikeGen(const Id& src, std::string name, Id& parent):SpikeGen(src, name, parent){}
StochSpikeGen::StochSpikeGen(const Id& src, std::string path):SpikeGen(src, path){}
StochSpikeGen::~StochSpikeGen(){}
const std::string& StochSpikeGen::getType(){ return className_; }
double StochSpikeGen::__get_failureP() const
{
    double failureP;
    get < double > (id_(), "failureP",failureP);
    return failureP;
}
void StochSpikeGen::__set_failureP( double failureP )
{
    set < double > (id_(), "failureP", failureP);
}
#endif
