#ifndef _pymoose_HHChannel_cpp
#define _pymoose_HHChannel_cpp
#include "pymoose.h"
const std::string HHChannel::className = "HHChannel";
HHChannel::HHChannel(Id id):PyMooseBase(id){}
HHChannel::HHChannel(std::string path):PyMooseBase(className, path){}
HHChannel::HHChannel(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HHChannel::HHChannel(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
HHChannel::~HHChannel(){}
const std::string& HHChannel::getType(){ return className; }
double HHChannel::__get_Gbar() const
{
    double Gbar;
    get < double > (Element::element(id_), "Gbar",Gbar);
    return Gbar;
}
void HHChannel::__set_Gbar( double Gbar )
{
    set < double > (Element::element(id_), "Gbar", Gbar);
}
double HHChannel::__get_Ek() const
{
    double Ek;
    get < double > (Element::element(id_), "Ek",Ek);
    return Ek;
}
void HHChannel::__set_Ek( double Ek )
{
    set < double > (Element::element(id_), "Ek", Ek);
}
double HHChannel::__get_Xpower() const
{
    double Xpower;
    get < double > (Element::element(id_), "Xpower",Xpower);
    return Xpower;
}
void HHChannel::__set_Xpower( double Xpower )
{
    set < double > (Element::element(id_), "Xpower", Xpower);
}
double HHChannel::__get_Ypower() const
{
    double Ypower;
    get < double > (Element::element(id_), "Ypower",Ypower);
    return Ypower;
}
void HHChannel::__set_Ypower( double Ypower )
{
    set < double > (Element::element(id_), "Ypower", Ypower);
}
double HHChannel::__get_Zpower() const
{
    double Zpower;
    get < double > (Element::element(id_), "Zpower",Zpower);
    return Zpower;
}
void HHChannel::__set_Zpower( double Zpower )
{
    set < double > (Element::element(id_), "Zpower", Zpower);
}
int HHChannel::__get_instant() const
{
    int instant;
    get < int > (Element::element(id_), "instant",instant);
    return instant;
}
void HHChannel::__set_instant( int instant )
{
    set < int > (Element::element(id_), "instant", instant);
}
double HHChannel::__get_Gk() const
{
    double Gk;
    get < double > (Element::element(id_), "Gk",Gk);
    return Gk;
}
void HHChannel::__set_Gk( double Gk )
{
    set < double > (Element::element(id_), "Gk", Gk);
}
double HHChannel::__get_Ik() const
{
    double Ik;
    get < double > (Element::element(id_), "Ik",Ik);
    return Ik;
}
void HHChannel::__set_Ik( double Ik )
{
    set < double > (Element::element(id_), "Ik", Ik);
}
int HHChannel::__get_useConcentration() const
{
    int useConcentration;
    get < int > (Element::element(id_), "useConcentration",useConcentration);
    return useConcentration;
}
void HHChannel::__set_useConcentration( int useConcentration )
{
    set < int > (Element::element(id_), "useConcentration", useConcentration);
}
double HHChannel::__get_IkSrc() const
{
    double IkSrc;
    get < double > (Element::element(id_), "IkSrc",IkSrc);
    return IkSrc;
}
void HHChannel::__set_IkSrc( double IkSrc )
{
    set < double > (Element::element(id_), "IkSrc", IkSrc);
}
double HHChannel::__get_concen() const
{
    double concen;
    get < double > (Element::element(id_), "concen",concen);
    return concen;
}
void HHChannel::__set_concen( double concen )
{
    set < double > (Element::element(id_), "concen", concen);
}


void HHChannel::tweakAlpha(std::string gate)
{
    this->getContext()->tweakAlpha(this->path(),gate);    
}

void HHChannel::tweakTau(std::string gate)
{
    this->getContext()->tweakTau(this->path(),gate);    
}

void HHChannel::setupAlpha(std::string gate, vector <double> params)
{
    this->getContext()->setupAlpha(this->path(),gate, params);    
}

void HHChannel::setupAlpha(std::string gate, double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max )
{
    // TODO:
}


void HHChannel::setupTau(std::string gate, vector <double> params)
{
    this->getContext()->setupTau(this->path(),gate, params);    
}
void HHChannel::setupTau(std::string gate, double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size, double min, double max)
{
    // TODO:
}

void HHChannel::createTable(std::string gate, unsigned int divs, double min, double max)
{
    if (gate.empty())
    {
        cerr << "Error: Gate cannot be empty std::string." << endl;
        return;        
    }
    
    if (gate.at(0)== 'X' || gate.at(0)== 'x')
    {
        gate = "xGate";        
    }
    else if (gate.at(0) == 'Y' || gate.at(0)== 'y' )
    {
        gate = "yGate";        
    }
    else if (gate.at(0) == 'Y' || gate.at(0)== 'y' )
    {
        gate = "zGate";        
    }
    else 
    {
        cerr << "Error: Gate must be one of X, Y or Z" << endl;
        return;        
    }
    std::string path = this->path()+"/"+gate + "/" +"A";

    Id id = PyMooseBase::pathToId(path);
    if ( id == 0 || id == BAD_ID )
    {
        cerr << "Error: " << " HHChannel::createTable(...) - Object does not exist: " << path << endl;
        return;        
    }
    InterpolationTable tableA(id);
    tableA.__set_xmin(min);
    tableA.__set_xmax(max);
    tableA.__set_xdivs(divs);
    path = this->path()+"/"+gate + "/" +"B";
    id = PyMooseBase::pathToId(path);
    if ( id == 0 || id == BAD_ID )
    {
        cerr << "Error: " << " HHChannel::createTable(...) - Object does not exist: " << path << endl;
        return;        
    }
    InterpolationTable tableB(id);    
    tableB.__set_xmin(min);
    tableB.__set_xmax(max);
    tableB.__set_xdivs(divs);
}

#endif
