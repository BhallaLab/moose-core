#ifndef _pymoose_Interpol_cpp
#define _pymoose_Interpol_cpp
#include "pymoose/Interpol.h"
#include "pymoose/TableIterator.h"

#ifdef _USE_NUMPY // this was experimental and the relevant  functions should be either here or in moose.i in the xtend block. not both.
#include <algorithm>
#include <Python.h>
#include <numpy/arrayobject.h>
#endif //! USE_NUMPY

using namespace pymoose;
// We had to change the class name in order to avoid conflict with included moose class Interpol
// But Interpol is the class name to be passed to base constructor
const std::string Interpol::className_ = "Interpol";
Interpol::Interpol(Id id):Neutral(id){}
Interpol::Interpol(std::string path):Neutral("Interpol", path){}
Interpol::Interpol(std::string name, Id parentId):Neutral("Interpol", name, parentId){}
Interpol::Interpol(
    std::string name,
    PyMooseBase& parent)
    :Neutral("Interpol", name, parent){}

Interpol::Interpol(
    const Interpol& src,
    std::string objectName,
    PyMooseBase& parent)
    :Neutral(src, objectName, parent){}

Interpol::Interpol(
    const Interpol& src,
    std::string objectName,
    Id& parent)
    :Neutral(src, objectName, parent){}

Interpol::Interpol(
    const Interpol& src,
    std::string path)
    :Neutral(src, path)
{
}
Interpol::Interpol(
    const Id& src,
    std::string path)
    :Neutral(src, path)
{
}

Interpol::Interpol(
    const Id& src,
    string name,
    Id& parent)
    :Neutral(src, name, parent)
{
}
Interpol::~Interpol(){}
//Manually edited
// These are for allowing Table access to constructors in PyMooseBase
Interpol::Interpol(std::string typeName, std::string objectName, Id parentId): Neutral(typeName, objectName, parentId)
{
}
   
Interpol::Interpol(std::string typeName, std::string path): Neutral(typeName, path)
{
}

Interpol::Interpol(std::string typeName, std::string objectName, PyMooseBase& parent): Neutral(typeName, objectName, parent)
{
}


const std::string& Interpol::getType(){ return className_; }
double Interpol::__get_xmin() const
{
    double xmin;
    get < double > (id_(), "xmin",xmin);
    return xmin;
}
void Interpol::__set_xmin( double xmin )
{
    set < double > (id_(), "xmin", xmin);
}
double Interpol::__get_xmax() const
{
    double xmax;
    get < double > (id_(), "xmax",xmax);
    return xmax;
}
void Interpol::__set_xmax( double xmax )
{
    set < double > (id_(), "xmax", xmax);
}
int Interpol::__get_xdivs() const
{
    int xdivs;
    get < int > (id_(), "xdivs",xdivs);
    return xdivs;
}
void Interpol::__set_xdivs( int xdivs )
{
    set < int > (id_(), "xdivs", xdivs);
}
int Interpol::__get_mode() const
{
    int mode;
    get < int > (id_(), "mode",mode);
    return mode;
}
void Interpol::__set_mode( int mode )
{
    set < int > (id_(), "mode", mode);
}
// int Interpol::__get_calc_mode() const
// {
//     int calc_mode;
//     get < int > (id_(), "calc_mode",calc_mode);
//     return calc_mode;
// }
// void Interpol::__set_calc_mode( int calc_mode )
// {
//     set < int > (id_(), "calc_mode", calc_mode);
// }
double Interpol::__get_dx() const
{
    double dx;
    get < double > (id_(), "dx", dx);
    return dx;
}
void Interpol::__set_dx( double dx )
{
    set < double > (id_(), "dx", dx);
}
double Interpol::__get_sy() const
{
    double sy;
    get < double > (id_(), "sy",sy);
    return sy;
}
void Interpol::__set_sy( double sy )
{
    set < double > (id_(), "sy", sy);
}
double Interpol::__getitem__(const unsigned int index) const
{
    double table;
    lookupGet <double, unsigned int>(id_.eref(), "table", table, index);
    return table;
}

void Interpol::__setitem__(unsigned int index, double value )
{
    lookupSet<double, unsigned int>(id_.eref(), "table", value, index);
}

TableIterator* Interpol::__iter__()
{
    return new TableIterator(this);    
}
int Interpol::__len__()
{
    return __get_xdivs()+1;    
}

int Interpol::__get_calcMode() const
{
    int calc_mode;
    get < int > (id_(), "calc_mode",calc_mode);
    return calc_mode;
}
void Interpol::__set_calcMode( int calc_mode )
{
    set < int > (id_(), "calc_mode", calc_mode);
}

string Interpol::dumpFile() const
{
    string print;
    get < string > (id_(), "print", print);
    return print;
}

void Interpol::dumpFile( string fileName, bool append )
{
    if (append)
    {
        set < string > ( id_(), "append", fileName);
    }
    else
    {
        set < string > (id_(), "print", fileName);
    }    
}

void Interpol::load(string fileName, unsigned int skiplines)
{
    set<string, unsigned int>(id_(), "load", fileName, skiplines);
}

void Interpol::tabFill(int xdivs, int mode)
{
    set <int, int> (id_(), "tabFill", xdivs, mode);    
}

void Interpol::clear()
{
    set(id_(), "clear");
}

void Interpol::push(double value)
{
    set<double>(id_(), "push", value);
}

void Interpol::pop()
{
    set(id_(), "pop");
}
const vector<double>& Interpol::__get_table()
{    
    return PyMooseBase::getContext()->getTableVector(id_);
}

#if 0
#ifdef _USE_NUMPY// this was experimental and the relevant  functions should be either here or in moose.i in the xtend block. not both.
PyObject* Interpol::__array_struct__()
{
    PyArrayObject* result;
    int dimensions[1];
    static vector <double> data;
    printf("In __array_struct__\n");
    data = Interpol::getTableVector(id_());
    dimensions[0] = data.size();
    // avoiding shared data copy - dangerous and the vector returned by getTableVector is a temporary copy 
    // result = (PyArrayObject*)PyArray_FromDimsAndData(1, dimensions, NPY_DOUBLE, (char*)(&data[0]));
    // instead create a PyArrayObject initialized with zero and then copy data
    result = (PyArrayObject*)PyArray_FromDims(1, dimensions, NPY_DOUBLE);
    memcpy(result->data, &data[0], dimensions[0]*sizeof(double));
    return PyArray_Return(result);
} // !__array_struct__

/**
   This function fills a table object using a Python sequence type object
*/

void Interpol::fillData(PyObject* args)
{
    PyObject* input_seq;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &input_seq))
        return;
    array = (PyArrayObject *) PyArray_ContiguousFromObject(input_seq, PyArray_DOUBLE, 1, 1);
    if (array == NULL)
        return;
    this->__set_xdivs(array->dimensions[0] - 1);
    vector <double> data(array->dimensions[0], 0.0);
    double min = *std::min_element(data.begin(), data.end());
    this->__set_xmin(min);
    double max = *std::max_element(data.begin(), data.end());
    this->__set_xmax(max);
    // todo: should use stl copy
    memcpy(&data[0], array->data, sizeof(double)*(array->dimensions[0]));
    set <vector <double> > (id_(), "tableVector", data);
}

#endif //! USE_NUMPY
#endif // !if 0
#ifdef DO_UNIT_TESTS
#include <cmath>
using namespace std;

void testPyMOOSEInterpol(void)
{
    Interpol table("/interptbl");
    double xmin = 0.0;
    double xmax = 10.0;
    int xdivs = 100;
    
    table.__set_xmin ( xmin );
    assert (fabs(table.__get_xmin() - xmin) < 1e-6 );
    
    table.__set_xmax ( xmax );
    assert (fabs(table.__get_xmax() - xmax) < 1e-6 );

    table.__set_xdivs ( xdivs );
    assert (table.__get_xdivs() == xdivs);
    
}

#endif // DO_UNIT_TESTS

#endif

