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
const std::string Cell::className = "Cell";
Cell::Cell(Id id):PyMooseBase(id){}
Cell::Cell(std::string path):PyMooseBase(className, path){}
Cell::Cell(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Cell::Cell(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Cell::~Cell(){}
const std::string& Cell::getType(){ return className; }
#endif
