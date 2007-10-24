/*******************************************************************
 * File:            Class.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
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
const std::string Class::className = "Class";
Class::Class(Id id):PyMooseBase(id){}
Class::Class(std::string path, std::string name):PyMooseBase(className, path)
{
    set <std::string> (id_(), "name", name);    
}
Class::Class(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Class::Class(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Class::~Class(){}
const std::string& Class::getType(){ return className; }

std::string Class::__get_name()
{
    std::string name;
    get <std::string> (id_(), "name", name);
    return name;    
}

void Class::__set_name(std::string name)
{
    set < std::string > ( id_(), "name", name);    
}

const std::string Class::__get_author()
{
    std::string author;
    get < std::string > ( id_(), "author", author);
    return author;
}

const std::string Class::__get_description()
{
    std::string description;
    get < std::string > ( id_(), "description", description);
    return description;
}

unsigned int Class::__get_tick()
{
    unsigned int tick;
    get <unsigned int > ( id_(), "tick", tick);
    return tick;
}

void Class::__set_tick(unsigned int tick)
{
    set < unsigned int > ( id_(), "tick", tick);    
}

unsigned int Class::__get_stage()
{
    unsigned int stage;
    get < unsigned int > ( id_(), "stage", stage);
    return stage;
}
void Class::__set_stage(unsigned int stage)
{
    set < unsigned int > (id_(), "stage", stage);
}

std::string Class::__get_clock()
{
    string clock;
    get < string > (id_(), "clock", clock);
    return clock;
}

void Class::setClock(std::string function, std::string clock)
{
    set < std::string, std::string > (id_(), "clock", function, clock);    
}

#endif
