#ifndef _pymoose_Interpol_cpp
#define _pymoose_Interpol_cpp
#include "Interpol.h"
#include "TableIterator.h"
#include "../builtins/Interpol.h"
using namespace pymoose;
// We had to change the class name in order to avoid conflict with included moose class Interpol
// But Interpol is the class name to be passed to base constructor
const std::string InterpolationTable::className = "InterpolationTable";
InterpolationTable::InterpolationTable(Id id):PyMooseBase(id){}
InterpolationTable::InterpolationTable(std::string path):PyMooseBase("Interpol", path){}
InterpolationTable::InterpolationTable(std::string name, Id parentId):PyMooseBase("Interpol", name, parentId){}
InterpolationTable::InterpolationTable(
    std::string name,
    PyMooseBase& parent)
    :PyMooseBase("Interpol", name, parent){}

InterpolationTable::InterpolationTable(
    const InterpolationTable& src,
    std::string objectName,
    PyMooseBase& parent)
    :PyMooseBase(src, objectName, parent){}

InterpolationTable::InterpolationTable(
    const InterpolationTable& src,
    std::string objectName,
    Id& parent)
    :PyMooseBase(src, objectName, parent){}

InterpolationTable::InterpolationTable(
    const InterpolationTable& src,
    std::string path)
    :PyMooseBase(src, path)
{
}

InterpolationTable::InterpolationTable(
    const Id& src,
    string name,
    Id& parent)
    :PyMooseBase(src, name, parent)
{
}
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

InterpolationTable::InterpolationTable(std::string typeName, std::string objectName, PyMooseBase& parent):
    PyMooseBase(typeName, objectName, parent)
{
}


const std::string& InterpolationTable::getType(){ return className; }
double InterpolationTable::__get_xmin() const
{
    double xmin;
    get < double > (id_(), "xmin",xmin);
    return xmin;
}
void InterpolationTable::__set_xmin( double xmin )
{
    set < double > (id_(), "xmin", xmin);
}
double InterpolationTable::__get_xmax() const
{
    double xmax;
    get < double > (id_(), "xmax",xmax);
    return xmax;
}
void InterpolationTable::__set_xmax( double xmax )
{
    set < double > (id_(), "xmax", xmax);
}
int InterpolationTable::__get_xdivs() const
{
    int xdivs;
    get < int > (id_(), "xdivs",xdivs);
    return xdivs;
}
void InterpolationTable::__set_xdivs( int xdivs )
{
    set < int > (id_(), "xdivs", xdivs);
}
int InterpolationTable::__get_mode() const
{
    int mode;
    get < int > (id_(), "mode",mode);
    return mode;
}
void InterpolationTable::__set_mode( int mode )
{
    set < int > (id_(), "mode", mode);
}
// int InterpolationTable::__get_calc_mode() const
// {
//     int calc_mode;
//     get < int > (id_(), "calc_mode",calc_mode);
//     return calc_mode;
// }
// void InterpolationTable::__set_calc_mode( int calc_mode )
// {
//     set < int > (id_(), "calc_mode", calc_mode);
// }
double InterpolationTable::__get_dx() const
{
    double dx;
    get < double > (id_(), "dx",dx);
    return dx;
}
void InterpolationTable::__set_dx( double dx )
{
    set < double > (id_(), "dx", dx);
}
double InterpolationTable::__get_sy() const
{
    double sy;
    get < double > (id_(), "sy",sy);
    return sy;
}
void InterpolationTable::__set_sy( double sy )
{
    set < double > (id_(), "sy", sy);
}
double InterpolationTable::__getitem__( unsigned int index) const
{
    double table;
    table = static_cast<Interpol*>(id_()->data())->getTableValue(index);    
    return table;
}
void InterpolationTable::__setitem__(unsigned int index, double value )
{
    static_cast<Interpol*>(id_()->data())->setTableValue(value, index);
}

TableIterator* InterpolationTable::__iter__()
{
    return new TableIterator(this);    
}
int InterpolationTable::__len__()
{
    return __get_xdivs()+1;    
}
#if 0
#ifdef NUMPY // Only for NumPy support
// TODO: work in progress
#include "numpy/noprefix.h"
PyObject* InterpolationTable::__array_struct__()
{
    PyArrayInterface* array = NULL;
    int dim = this->__get_xdivs() + 1;
    if ( dim > 1 )
    {  
        array = (PyArrayInterface*)PyArray_malloc(sizeof(PyArrayInterface));
        array->two = 2;
        array->nd = 1;
        array->typekind = 'f';
        array->itemsize = sizeof(double);
        array->flags = (NPY_CONTIGUOUS | NPY_OWNDATA | NPY_ALIGNED | NPY_NOTSWAPPED); 
        array->shape = (intp *)_pya_malloc(2*sizeof(intp));        
        array->strides = array->shape + 1;
        *(array->shape) = dim;
        *(array->strides) = 1;
        array->data = (char *)calloc(dim,sizeof(double));
        vector <double> data = static_cast < Interpol*> (id_()->data())->getTableVector(id_());
//        get < vector <double> > (this->id_(), "tableVector", data); // obtain the vector of data from InterpolTable
        
        memcpy(&data[0], array->data, sizeof(double)*(data.size())); // copy data from table to array obj
        for ( int i = 0; i < data.size(); ++i )
        {
            cout << data[i] << "\t" << (double)((double*)(array->data))[i] << endl;
        }
        
        array->descr = 0;
       
    }
    return PyCObject_FromVoidPtr(array, 0);
}
#endif // NUMPY
#endif // !commented out
int InterpolationTable::__get_calcMode() const
{
    int calc_mode;
    get < int > (id_(), "calc_mode",calc_mode);
    return calc_mode;
}
void InterpolationTable::__set_calcMode( int calc_mode )
{
    set < int > (id_(), "calc_mode", calc_mode);
}

string InterpolationTable::dumpFile() const
{
    string print;
    get < string > (id_(), "print", print);
    return print;
}

void InterpolationTable::dumpFile( string fileName, bool append )
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

void InterpolationTable::tabFill(int xdivs, int mode)
{
    set <int, int> (id_(), "tabFill", xdivs, mode);    
}

#ifdef DO_UNIT_TESTS
#include <cmath>
using namespace std;

void testInterpolationTable(void)
{
    InterpolationTable table("/interptbl");
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

