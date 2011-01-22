#ifndef _pymoose_Interpol2D_cpp
#define _pymoose_Interpol2D_cpp

#include "pymoose/Interpol2D.h"
using namespace pymoose;
const std::string Interpol2D::className_ = "Interpol2D";
Interpol2D::Interpol2D(Id id):Interpol(id){}
Interpol2D::Interpol2D(std::string path):Interpol(className_, path){}
Interpol2D::Interpol2D(std::string name, Id parentId):Interpol(className_, name, parentId){}
Interpol2D::Interpol2D(std::string name, PyMooseBase& parent):Interpol(className_, name, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string objectName, PyMooseBase& parent):Interpol(src, objectName, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string objectName, Id& parent):Interpol(src, objectName, parent){}
Interpol2D::Interpol2D(const Interpol2D& src, std::string path):Interpol(src, path){}
Interpol2D::Interpol2D(const Id& src, std::string name, Id& parent):Interpol(src, name, parent){}
Interpol2D::Interpol2D(const Id& src, std::string path):Interpol(src, path){}
Interpol2D::~Interpol2D(){}
const std::string& Interpol2D::getType(){ return className_; }
double Interpol2D::__get_ymin() const
{
    double ymin;
    get < double > (id_(), "ymin",ymin);
    return ymin;
}
void Interpol2D::__set_ymin( double ymin )
{
    set < double > (id_(), "ymin", ymin);
}
double Interpol2D::__get_ymax() const
{
    double ymax;
    get < double > (id_(), "ymax",ymax);
    return ymax;
}
void Interpol2D::__set_ymax( double ymax )
{
    set < double > (id_(), "ymax", ymax);
}
int Interpol2D::__get_ydivs() const
{
    int ydivs;
    get < int > (id_(), "ydivs",ydivs);
    return ydivs;
}
void Interpol2D::__set_ydivs( int ydivs )
{
    set < int > (id_(), "ydivs", ydivs);
}
double Interpol2D::__get_dy() const
{
    double dy;
    get < double > (id_(), "dy",dy);
    return dy;
}
void Interpol2D::__set_dy( double dy )
{
    set < double > (id_(), "dy", dy);
}

double Interpol2D::getTableValue(unsigned int row, unsigned int col)
{
    vector<unsigned int> index;
    index.push_back(row);
    index.push_back(col);
    double value;
    // Minor bug in LookupFinfo::match - if I put table2D instead, I
    // get an error: Error: LookupFinfo::match: bad indexing: table2D
    // as the function matches the 'table' part and assumes 2D to be
    // an index. It should instead check for '['.
    lookupGet< double, vector< unsigned int > >(id_(), "table", value, index);
    return value;
}

void Interpol2D::setTableValue(unsigned int row, unsigned int col, double value)
{
    vector<unsigned int> index;
    index.push_back(row);
    index.push_back(col);
    lookupSet< double, vector< unsigned int > >(id_(), "table", value, index);
}

#endif
