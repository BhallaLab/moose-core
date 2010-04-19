#ifndef _pymoose_NeuroScan_cpp
#define _pymoose_NeuroScan_cpp
#include "NeuroScan.h"
using namespace pymoose;
const std::string NeuroScan::className_ = "NeuroScan";
NeuroScan::NeuroScan(Id id):PyMooseBase(id){}
NeuroScan::NeuroScan(std::string path):PyMooseBase(className_, path){}
NeuroScan::NeuroScan(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
NeuroScan::NeuroScan(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
NeuroScan::NeuroScan(const NeuroScan& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
NeuroScan::NeuroScan(const NeuroScan& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
NeuroScan::NeuroScan(const NeuroScan& src, std::string path):PyMooseBase(src, path){}
NeuroScan::NeuroScan(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
NeuroScan::NeuroScan(const Id& src, std::string path):PyMooseBase(src, path){}

NeuroScan::~NeuroScan(){}
const std::string& NeuroScan::getType(){ return className_; }
int NeuroScan::__get_VDiv() const
{
    int VDiv;
    get < int > (id_(), "VDiv",VDiv);
    return VDiv;
}
void NeuroScan::__set_VDiv( int VDiv )
{
    set < int > (id_(), "VDiv", VDiv);
}
double NeuroScan::__get_VMin() const
{
    double VMin;
    get < double > (id_(), "VMin",VMin);
    return VMin;
}
void NeuroScan::__set_VMin( double VMin )
{
    set < double > (id_(), "VMin", VMin);
}
double NeuroScan::__get_VMax() const
{
    double VMax;
    get < double > (id_(), "VMax",VMax);
    return VMax;
}
void NeuroScan::__set_VMax( double VMax )
{
    set < double > (id_(), "VMax", VMax);
}
int NeuroScan::__get_CaDiv() const
{
    int CaDiv;
    get < int > (id_(), "CaDiv",CaDiv);
    return CaDiv;
}
void NeuroScan::__set_CaDiv( int CaDiv )
{
    set < int > (id_(), "CaDiv", CaDiv);
}
double NeuroScan::__get_CaMin() const
{
    double CaMin;
    get < double > (id_(), "CaMin",CaMin);
    return CaMin;
}
void NeuroScan::__set_CaMin( double CaMin )
{
    set < double > (id_(), "CaMin", CaMin);
}
double NeuroScan::__get_CaMax() const
{
    double CaMax;
    get < double > (id_(), "CaMax",CaMax);
    return CaMax;
}
void NeuroScan::__set_CaMax( double CaMax )
{
    set < double > (id_(), "CaMax", CaMax);
}
#endif
