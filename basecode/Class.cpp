/*******************************************************************
 * File:            Class.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-13 21:10:43
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

#ifndef _CLASS_CPP
#define _CLASS_CPP

#include "Class.h"
#include <cstdlib>
#include "Cinfo.h"
// #include "Conn.h"
// #include "Element.h"

const Cinfo* initClassCinfo()
{
    static Finfo* classFinfos[] = 
        {
            new ValueFinfo( "name", ValueFtype1< string >::global(),
                            GFCAST ( &Class::getName),
                            RFCAST(&Class::setName)
                ),
            new ValueFinfo( "fields",
                            ValueFtype1 < vector <string> >::global(),
                            GFCAST ( &Class::getFieldList ),
                            &dummyFunc
                ),
            
            new ValueFinfo( "author",
                            ValueFtype1 < string >::global(),
                            GFCAST ( &Class::getAuthor),
                            &dummyFunc
                ),
            new ValueFinfo( "description", ValueFtype1 < string >::global(),
                            GFCAST ( &Class::getDescription),
                            &dummyFunc
                ),
            new ValueFinfo( "tick",
                            ValueFtype1 < unsigned int >::global(),
                            GFCAST ( &Class::getTick ),
                            RFCAST ( &Class::setTick )
                ),
            new ValueFinfo( "stage", ValueFtype1 < unsigned int >::global(),
                            GFCAST ( &Class::getStage ),
                            RFCAST ( &Class::setStage )
                ),
            new DestFinfo( "clock",
                           Ftype2 < string, Id >::global(),
                           RFCAST(Class::setClock),
						   "Schedule function (string) on clock tick ( Id )"
                ),
//             new ValueFinfo( "base", ValueFtype1 <string>::global(),
//                             reinterpret_cast < GetFunc > ( &Class::getBase ),
//                             &dummyFunc
//                 ), 
        };
    static Cinfo classCinfo( "Class",
                             "Subhasis Ray",
                             "Class object. Meta-information for MOOSE classes.",
                             initNeutralCinfo(),
                             classFinfos,
                             sizeof(classFinfos)/sizeof(Finfo*),
                             ValueFtype1 < Class >::global()
        );
    return &classCinfo;    
}

static const Cinfo * classCinfo = initClassCinfo();


Class::Class(string name)
{
    map < string, Cinfo* >::iterator i = Cinfo::lookup().find(name);
    if ( i != Cinfo::lookup().end())
    {
        classInfo_ = i->second;
    }
    else
    {
        classInfo_ = 0;
        cerr << "ERROR: " << name << " no such class exists." << endl;
    }    
}

const string Class::getName(const Element* e)
{
    Class* obj = static_cast<Class*>(e->data( 0 ));
    if ( obj->classInfo_)
    {
        return obj->classInfo_->name();
    }
    else
    {
        return "";
    }    
}
void Class::setName(const Conn* conn, string name)
{
    Class* obj = static_cast<Class*> (conn->data());
    
    map < string, Cinfo* >::iterator i = Cinfo::lookup().find(name);
    if ( i != Cinfo::lookup().end())
    {
        obj->classInfo_ = i->second;
    }
    else
    {
        obj->classInfo_ = 0;
        cerr << "ERROR: " << name << " no such class exists." << endl;
    }
}
/// Set the SchedInfo for the class such that it is scheduled to clock specified by tickId
void Class::setClock(const Conn* conn, string function, Id tickId)
{
    assert (!tickId.zero());
    string tickPath = tickId.path();
    unsigned int clockNo = atoi(tickPath.substr(tickPath.find_last_of('t')).c_str()); // assuming all clock tick paths are tN where N is the clock no.
    unsigned int tick = clockNo / 2;
    unsigned int stage = clockNo % 2;
    
    Class* obj = static_cast<Class*>(conn->data());
    SchedInfo* schedInfo;
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return;
    }
    
    ///\TODO: trouble trouble - SchedInfo is magically const ...
    for ( unsigned int i = 0; i < obj->classInfo_->scheduling_.size(); ++ i )
    {
        schedInfo = &(obj->classInfo_->scheduling_[i]);
        
        const Finfo* finfo = schedInfo->finfo;
        if ( function == finfo->name() )
        {
            schedInfo->tick = tick;
            schedInfo->stage = stage;
            return;            
        }
    }
    // No scheduling information for the class, insert one
    // but that is not possible from here ...
//     SchedInfo schedInfo;
//     schedInfo.finfo = new SharedFinfo(function, fnSharedFinfoArray, size(fnSharedFinfoArray)/sizeof();
    
}


/// Returns the tick of the process SchedInfo
unsigned int Class::getTick(const Element* e)
{
    Class* obj = static_cast<Class*>(e->data( 0 ));
    
    string name = "process";    
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return BAD_TICK;
    }
    for ( vector< struct SchedInfo >::const_iterator i = obj->classInfo_->scheduling_.begin();
          i != obj->classInfo_->scheduling_.end(); ++i )
    {
        if ( name == i->finfo->name() )
        {
            return i->tick;            
        }
    }        
    
    return BAD_TICK;        
}

/// Sets the tick no. of the process SchedInfo
void Class::setTick(const Conn* conn, unsigned int tick)
{
    string name = "process";
    Class * obj = static_cast<Class*> (conn->data());
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return;
    }
    for (vector<struct SchedInfo>::iterator i = obj->classInfo_->scheduling_.begin();
         i != obj->classInfo_->scheduling_.end();
         ++i)
    {
        const Finfo* finfo = i->finfo;
        if ( name == finfo->name())
        {
            // TODO: validate the tick and stage here?
            i->tick = tick;
        }
    }
}

/// Returns the stage of the process SchedInfo
unsigned int Class::getStage(const Element* e)
{
    string name = "process";
    Class* obj = static_cast<Class*> (e->data( 0 ));
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return 0;
    }
    for ( vector< struct SchedInfo >::iterator i = obj->classInfo_->scheduling_.begin();
          i != obj->classInfo_->scheduling_.end(); ++i )
    {
        if ( name == i->finfo->name() )
        {
            return i->stage;            
        }
    }    
    return BAD_STAGE;           
}


/// Set the stage of the process function in SchedInfo
void Class::setStage(const Conn* conn, unsigned int stage)
{
    string name = "process";
    Class* obj = static_cast<Class*>(conn->data());
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return;
    }
    for (vector<struct SchedInfo>::iterator i = obj->classInfo_->scheduling_.begin();
         i != obj->classInfo_->scheduling_.end();
         ++i )
    {
        const Finfo* finfo = i->finfo;
        if ( name == finfo->name())
        {
            // TODO: validate the tick and stage here?
            i->stage = stage;            
        }
    }
}

/// Returns a list of Finfos that are part of this class
vector <string> Class::getFieldList(const Element* e)
{
    vector <string> fieldList;
    Class *obj = static_cast<Class*> (e->data( 0 ));
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
    }
    else 
    {
        for (unsigned int i = 0; i < obj->classInfo_->finfos_.size() ; ++i )
        {
            Finfo *f = obj->classInfo_->finfos_[i];
            fieldList.push_back(f->name());
        }
    }
    
    return fieldList;
    
}

/// Returns the name of the author of the class as passed to the Cinfo constructor.
const string Class::getAuthor(const Element* e)
{
    Class * obj = static_cast<Class*> ( e->data( 0 ) );
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return "";
    }
    return obj->classInfo_->author();    
}

/// Returns the description of the Class as passed to the Cinfo constructor
const string Class::getDescription(const Element* e)
{
    Class * obj = static_cast<Class* > ( e->data( 0 ) );
    
    if (!obj->classInfo_)
    {
        cerr << "Error: Class object not initialized." << endl;
        return "";
    }
    return obj->classInfo_->description();
}
// TODO: baseCinfo_ is not a member of Cinfo, it does not hold any info about base
// string Called::getBase()
// {
//     return classInfo_->baseCinfo_->name();
// }


// TODO: think about this
// void Class::setField(string name)
// {
// }

#endif
