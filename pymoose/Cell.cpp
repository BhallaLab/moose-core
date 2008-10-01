/*******************************************************************
 * File:            pymoose/Cell.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-24 15:56:43
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _pymoose_Cell_cpp
#define _pymoose_Cell_cpp
#include "Cell.h"
using namespace pymoose;
const std::string Cell::className = "Cell";
Cell::Cell(Id id):PyMooseBase(id){}
Cell::Cell(std::string path):PyMooseBase(className, path){}
Cell::Cell(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Cell::Cell(std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
Cell::Cell(const Cell& src, std::string objectName,  PyMooseBase& parent):PyMooseBase(src, objectName, parent){}

Cell::Cell(const Cell& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Cell::Cell(const Cell& src, std::string path):PyMooseBase(src, path)
{
}

Cell::Cell(const Id& src, string name, Id& parent):PyMooseBase(src, name, parent)
{
}
Cell::~Cell(){}
const std::string& Cell::getType(){ return className; }
string Cell::__get_method() const
{
    string method;
    get < string > (id_(), "method",method);
    return method;
}
void Cell::__set_method( string method )
{
    set < string > (id_(), "method", method);
}
bool Cell::__get_variableDt() const
{
    bool variableDt;
    get < bool > (id_(), "variableDt",variableDt);
    return variableDt;
}
bool Cell::__get_implicit() const
{
    bool implicit;
    get < bool > (id_(), "implicit",implicit);
    return implicit;
}
const string Cell::__get_description() const
{
    string description;
    get < string > (id_(), "description",description);
    return description;
}

#endif
