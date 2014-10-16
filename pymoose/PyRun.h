// PyRun.h --- 
// 
// Filename: PyRun.h
// Description: 
// Author: subha
// Maintainer: 
// Created: Sat Oct 11 14:40:45 2014 (+0530)
// Version: 
// Last-Updated: 
//           By: 
//     Update #: 0
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Class to call Python functions from MOOSE
// 
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#ifndef _PYCALL_H
#define _PYCALL_H

/**
   PyRun allows caling Python functions from moose.
 */
class PyRun
{
public:
    PyRun();
    ~PyRun();
    
    void setInitString(string str);
    string getInitString() const;

    void setRunString(string str);
    string getRunString() const;

    void setGlobals(PyObject *globals);
    PyObject * getGlobals() const;
    
    void setLocals(PyObject *locals);
    PyObject * getLocals() const;

    void setDebug(bool flag);
    bool getDebug() const;
    
    void run(string statement);

    void trigger(const Eref& e, double input); // this is a way to trigger execution via incoming message - can be useful for debugging
    
    void process(const Eref& e, ProcPtr p);
    void reinit(const Eref& e, ProcPtr p);
    
    static const Cinfo * initCinfo();

protected:
    bool debug_; // flag to enable debug output
    string initstr_; // statement str for running at reinit
    string runstr_; // statement str for running in each process call
    PyObject * globals_; // global env dict
    PyObject * locals_; // local env dict
    PyCodeObject * runcompiled_; // compiled form of procstr_
    PyCodeObject * initcompiled_; // coimpiled form of initstr_
};

#endif



// 
// PyRun.h ends here
