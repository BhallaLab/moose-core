// pymoose_Neutral.h --- 
// 
// Filename: pymoose.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Fri Mar 11 09:49:33 2011 (+0530)
// Version: 
// Last-Updated: Fri Mar 25 13:38:47 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 145
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 2011-03-11 09:50:06 (+0530) Dividing C++ wrapper and C externals
// for pymoose.
// 

// Code:
#ifndef _PYMOOSE_NEUTRAL_H
#define _PYMOOSE_NEUTRAL_H
#include <string>
#include "pymoose.h"

class Shell;
class Id;

namespace pymoose{
/**
 * Wraps Neutral.
 */
class pymoose_Neutral: public PyMooseBase
{
  public:
    /**
       Creates an instance with a given Id, useful for wrapping
       existing objets.
    */
    pymoose_Neutral(Id id);
    /**
       Create an element with given path and dimensions, type will be
       the class of the element.
     */
    pymoose_Neutral(string path, string type, vector<unsigned int> dims);
    /**
       Destructor does not do anything now.
     */
    ~pymoose_Neutral();
    /**
       Destroy the underlying MOOSE element.
    */
    int destroy();
    /**
       Return a copy of the Id of this object
    */
    Id id();
    /**
       Get the specified field of the index-th object in this element.
       The memory for the field is dynamically allocated and it is the
       responsibility of the caller to invoke the correct deallocation
       method.

       The shorttype of the field is returned in the ftype parameter.
    */
    void * getField(string fname, char& ftype, unsigned int index);
    /**
       Set the specified field's value on object at index.
     */
    int setField(string fname, void * value, unsigned int index);
    /**
       Get a vector containing names of all the fields of a specified kind.

       The kind can be any of:
               sourceFinfo
               destFinfo
               valueFinfo
               lookupFinfo
               sharedFinfo
    */
    vector<string> getFieldNames(string ftypeType);
    /**
       Returns the type of the specified field on this element.
     */
    string getFieldType(string field);
    /**
       Returns a list of Ids of all the children of the object at
       specified index.
    */
    vector<Id> getChildren(unsigned int index);
    /**
       addmsg equivalent
    */
    bool connect(string sourceField, pymoose_Neutral& dest, string destField, string msgType, unsigned int srcIndex, unsigned int destIndex, unsigned int srcDataIndex, unsigned int destDataIndex);
  protected:
    /**
       id_ is the sole reference to MOOSE elements.
    */
    Id id_;
};

} // ! namespace pymoose

#endif // !_PYMOOSE_NEUTRAL_H
// 
// pymoose_Neutral.h ends here
