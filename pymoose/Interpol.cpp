#ifndef _pymoose_Interpol_cpp
#define _pymoose_Interpol_cpp
#include "Interpol.h"
#include "TableIterator.h"
#include "../builtins/Interpol.h"
// We had to change the class name in order to avoid conflict with included moose class Interpol
// But Interpol is the class name to be passed to base constructor
const std::string InterpolationTable::className = "InterpolationTable";
InterpolationTable::InterpolationTable(Id id):PyMooseBase(id){}
InterpolationTable::InterpolationTable(std::string path):PyMooseBase("Interpol", path){}
InterpolationTable::InterpolationTable(std::string name, Id parentId):PyMooseBase("Interpol", name, parentId){}
InterpolationTable::InterpolationTable(std::string name, PyMooseBase* parent):PyMooseBase("Interpol", name, parent){}
InterpolationTable::~InterpolationTable(){}
//Manually edited
// These are for allowing Table access to constructors in PyMooseBase
InterpolationTable::InterpolationTable(std::string typeName, std::string objectName, Id parentId):
    PyMooseBase(typeName, objectName, parentId)
{
}
   
InterpolationTable::InterpolationTable(std::string typeName, std::string path):
    PyMooseBase(typeName, path)
{
}

InterpolationTable::InterpolationTable(std::string typeName, std::string objectName, PyMooseBase* parent):
    PyMooseBase(typeName, objectName, parent)
{
}


const std::string& InterpolationTable::getType(){ return className; }
double InterpolationTable::__get_xmin() const
{
    double xmin;
    get < double > (Element::element(id_), "xmin",xmin);
    return xmin;
}
void InterpolationTable::__set_xmin( double xmin )
{
    set < double > (Element::element(id_), "xmin", xmin);
}
double InterpolationTable::__get_xmax() const
{
    double xmax;
    get < double > (Element::element(id_), "xmax",xmax);
    return xmax;
}
void InterpolationTable::__set_xmax( double xmax )
{
    set < double > (Element::element(id_), "xmax", xmax);
}
int InterpolationTable::__get_xdivs() const
{
    int xdivs;
    get < int > (Element::element(id_), "xdivs",xdivs);
    return xdivs;
}
void InterpolationTable::__set_xdivs( int xdivs )
{
    set < int > (Element::element(id_), "xdivs", xdivs);
}
int InterpolationTable::__get_mode() const
{
    int mode;
    get < int > (Element::element(id_), "mode",mode);
    return mode;
}
void InterpolationTable::__set_mode( int mode )
{
    set < int > (Element::element(id_), "mode", mode);
}
int InterpolationTable::__get_calc_mode() const
{
    int calc_mode;
    get < int > (Element::element(id_), "calc_mode",calc_mode);
    return calc_mode;
}
void InterpolationTable::__set_calc_mode( int calc_mode )
{
    set < int > (Element::element(id_), "calc_mode", calc_mode);
}
double InterpolationTable::__get_dx() const
{
    double dx;
    get < double > (Element::element(id_), "dx",dx);
    return dx;
}
void InterpolationTable::__set_dx( double dx )
{
    set < double > (Element::element(id_), "dx", dx);
}
double InterpolationTable::__get_sy() const
{
    double sy;
    get < double > (Element::element(id_), "sy",sy);
    return sy;
}
void InterpolationTable::__set_sy( double sy )
{
    set < double > (Element::element(id_), "sy", sy);
}
double InterpolationTable::__getitem__( unsigned int index) const
{
    double table;
    table = static_cast<Interpol*>(Element::element(id_)->data())->getTableValue(index);    
    return table;
}
void InterpolationTable::__setitem__(unsigned int index, double value )
{
    static_cast<Interpol*>(Element::element(id_)->data())->setTableValue(value, index);
}

TableIterator* InterpolationTable::__iter__()
{
    return new TableIterator(this);    
}
int InterpolationTable::__len__()
{
    return __get_xdivs()+1;    
}

double InterpolationTable::__get_lookupSrc() const
{
    double lookupSrc;
    get < double > (Element::element(id_), "lookupSrc",lookupSrc);
    return lookupSrc;
}
void InterpolationTable::__set_lookupSrc( double lookupSrc )
{
    set < double > (Element::element(id_), "lookupSrc", lookupSrc);
}
double InterpolationTable::__get_lookup() const
{
    double lookup;
    get < double > (Element::element(id_), "lookup",lookup);
    return lookup;
}
void InterpolationTable::__set_lookup( double lookup )
{
    set < double > (Element::element(id_), "lookup", lookup);
}
string InterpolationTable::dumpFile() const
{
    string print;
    get < string > (Element::element(id_), "print",print);
    return print;
}
void InterpolationTable::dumpFile( string fileName )
{
    set < string > (Element::element(id_), "print", fileName);
}
/**
   What are the possible values for mode?
 */
void InterpolationTable::tabFill( int xdivs, int mode )
{
    this->getContext()->tabFill(id_, xdivs, mode);    
}

#endif
