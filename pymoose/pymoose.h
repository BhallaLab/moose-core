// pymoose.h --- 
// 
// Filename: pymoose.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Fri Mar 11 09:49:33 2011 (+0530)
// Version: 
// Last-Updated: Wed Feb 15 14:06:37 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 200
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
#ifndef _PYMOOSE_H
#define _PYMOOSE_H
#include <string>
#include <map>

class Shell;
class Id;
namespace pymoose{
/**
 * Base of the whole PyMoose class hierarchy.
 */
class PyMooseBase
{
  public:
    PyMooseBase();
    virtual ~PyMooseBase();    
}; // ! class PyMooseBase

////////////////////////////////////////////////////////
// These functions are defined in pymooseutil.cpp
////////////////////////////////////////////////////////

void setup_runtime_env(bool verbose=true);
Shell& getShell(int argc=0, char ** argv=0);
void finalize();
string getFieldType(ObjId id, string fieldName, string finfoType);
pair<string, string> getFieldFinfoTypePair(ObjId id, string fieldName);

vector<string> getFieldNames(ObjId id, string finfoType);

} // ! namespace pymoose

        
#endif // !_PYMOOSE_H
// 
// pymoose.h ends here
