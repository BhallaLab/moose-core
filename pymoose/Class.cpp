/*******************************************************************
 * File:            Class.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-10-24 16:00:11
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
#ifndef _pymoose_Class_cpp
#define _pymoose_Class_cpp

#include "Class.h"
using namespace pymoose;
const std::string pymoose::Class::className_ = "Class";
pymoose::Class::Class(Id id):Neutral(id){}
pymoose::Class::Class(std::string path, std::string name):Neutral(className_, path)
{
    set <std::string> (id_(), "name", name);    
}
pymoose::Class::Class(std::string name, Id parentId):Neutral(className_, name, parentId){}
pymoose::Class::Class(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
pymoose::Class::Class(const pymoose::Class& src, std::string objectName,  PyMooseBase& parent):Neutral(src, objectName, parent){}

pymoose::Class::Class(const pymoose::Class& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
pymoose::Class::Class(const pymoose::Class& src, std::string path):Neutral(src, path)
{
}
pymoose::Class::Class(const Id& src, std::string path):Neutral(src, path)
{
}

pymoose::Class::Class(const Id& src, string name, Id& parent):Neutral(src, name, parent)
{
}
pymoose::Class::~Class(){}
const std::string& pymoose::Class::getType(){ return className_; }

std::string pymoose::Class::__get_name()
{
    std::string name;
    get <std::string> (id_(), "name", name);
    return name;    
}

void pymoose::Class::__set_name(std::string name)
{
    set < std::string > ( id_(), "name", name);    
}

const std::string pymoose::Class::__get_author()
{
    std::string author;
    get < std::string > ( id_(), "author", author);
    return author;
}

const std::string pymoose::Class::__get_description()
{
    std::string description;
    get < std::string > ( id_(), "description", description);
    return description;
}

unsigned int pymoose::Class::__get_tick()
{
    unsigned int tick;
    get <unsigned int > ( id_(), "tick", tick);
    return tick;
}

void pymoose::Class::__set_tick(unsigned int tick)
{
    set < unsigned int > ( id_(), "tick", tick);    
}

unsigned int pymoose::Class::__get_stage()
{
    unsigned int stage;
    get < unsigned int > ( id_(), "stage", stage);
    return stage;
}
void pymoose::Class::__set_stage(unsigned int stage)
{
    set < unsigned int > (id_(), "stage", stage);
}

std::string pymoose::Class::__get_clock()
{
    string clock;
    get < string > (id_(), "clock", clock);
    return clock;
}

void pymoose::Class::setClock(std::string function, std::string clock)
{
    set < std::string, std::string > (id_(), "clock", function, clock);    
}

#endif
