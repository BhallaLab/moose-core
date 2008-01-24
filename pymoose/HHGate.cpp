#ifndef _pymoose_HHGate_cpp
#define _pymoose_HHGate_cpp
#include "HHGate.h"
using namespace pymoose;
const std::string HHGate::className = "HHGate";
HHGate::HHGate(Id id):PyMooseBase(id){}
HHGate::HHGate(std::string path):PyMooseBase(className, path){}
HHGate::HHGate(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HHGate::HHGate(std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
HHGate::~HHGate(){}
const std::string& HHGate::getType(){ return className; }

// Manually edited part
InterpolationTable* HHGate::__get_A() const
{
    return new InterpolationTable(PyMooseBase::pathToId(this->path()+"/A"));    
}
InterpolationTable* HHGate::__get_B() const
{
    return new InterpolationTable(PyMooseBase::pathToId(this->path()+"/B"));
}

void HHGate::tabFill(int xdivs, int mode)
{
    this->__get_A()->tabFill(xdivs, mode);
    this->__get_B()->tabFill(xdivs, mode);
}
void HHGate::setupAlpha(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max)
{
    std::string mypath = this->path();
    std::string::size_type separatorIndex = mypath.find(getSeparator());
    if (separatorIndex==std::string::npos)
    {
        cerr << "Error: Gate is not contained inside a channel." << endl;
        return;        
    }    
    std::string channel = mypath.substr(0,separatorIndex-1);
    std::string myName = mypath.substr(separatorIndex);
    
    this->getContext()->setupAlpha(channel, myName,AA, AB, AC, AD, AF, BA, BB, BC, BD, BF, size, min, max);    
}

void HHGate::setupTau(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max)
{
    
    std::string mypath = this->path();
    std::string::size_type separatorIndex = mypath.find(getSeparator());
    if (separatorIndex==std::string::npos)
    {
        cerr << "Error: Gate is not contained inside a channel." << endl;
        return;        
    }    
    std::string channel = mypath.substr(0,separatorIndex-1);
    std::string myName = mypath.substr(separatorIndex);
    
    this->getContext()->setupAlpha(channel,myName, AA, AB, AC, AD, AF, BA, BB, BC, BD, BF, size, min, max);    
}



#endif
