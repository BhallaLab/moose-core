#ifndef _pymoose_HHGate_cpp
#define _pymoose_HHGate_cpp
#include "HHGate.h"
const std::string HHGate::className = "HHGate";
HHGate::HHGate(Id id):PyMooseBase(id){}
HHGate::HHGate(std::string path):PyMooseBase(className, path){}
HHGate::HHGate(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HHGate::HHGate(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
HHGate::~HHGate(){}
const std::string& HHGate::getType(){ return className; }

// Manually edited part
Table* HHGate::getA() const
{
    return new Table(PyMooseBase::pathToId(this->path()+"/A"));    
}
Table* HHGate::getB() const
{
    return new Table(PyMooseBase::pathToId(this->path()+"/B"));
}

void HHGate::tabFill(int xdivs, int mode)
{
    this->getA()->tabFill(xdivs, mode);
    this->getB()->tabFill(xdivs, mode);
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
