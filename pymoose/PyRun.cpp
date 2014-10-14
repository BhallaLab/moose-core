// PyRun.cpp --- 
// 
// Filename: PyRun.cpp
// Description: 
// Author: subha
// Maintainer: 
// Created: Sat Oct 11 14:47:22 2014 (+0530)
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
// 
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

#include "Python.h"
#include "../basecode/header.h"
#include "PyRun.h"

const Cinfo * PyRun::initCinfo()
{
    static ValueFinfo< PyRun, string > pystring(
        "string",
        "String to be executed.",
        &PyRun::setString,
        &PyRun::getString);

    // static ValueFinfo< PyRun, PyObject* > globals(
    //     "globals",
    //     "Global environment dict",
    //     &PyRun::setGlobals,
    //     &PyRun::getGlobals);

    // static ValueFinfo< PyRun, PyObject* > locals(
    //     "locals",
    //     "Local environment dict",
    //     &PyRun::setLocals,
    //     &PyRun::getLocals);
    
    static DestFinfo run(
        "run",
        "Runs a specified string. Does not modify existing python statement.",
        new OpFunc1< PyRun, string >(&PyRun::run));

    static DestFinfo process(
        "process",
        "Handles process call. Runs the current string. This is not suitable for print statements.",
        new ProcOpFunc< PyRun >(&PyRun::process));

    static DestFinfo reinit(
        "reinit",
        "Handles reinit call.",
        new ProcOpFunc< PyRun >( &PyRun::reinit ));
    
    static Finfo * processShared[] = { &process, &reinit };
    static SharedFinfo proc(
        "proc",
        "This is a shared message to receive Process messages "
        "from the scheduler objects."
        "The first entry in the shared msg is a MsgDest "
        "for the Process operation. It has a single argument, "
        "ProcInfo, which holds lots of information about current "
        "time, thread, dt and so on. The second entry is a MsgDest "
        "for the Reinit operation. It also uses ProcInfo. ",
        processShared, sizeof( processShared ) / sizeof( Finfo* ));

    static Finfo * pyRunFinfos[] = {
        &pystring,
        // &locals,
        // &globals,
        &run,
        &proc,
    };

    static string doc[] = {
        "Name", "PyRun",
        "Author", "Subhasis Ray",
        "Description", "Runs Python statements from inside MOOSE."};
    static Dinfo< PyRun > dinfo;
    static Cinfo pyRunCinfo(
        "PyRun",
        Neutral::initCinfo(),
        pyRunFinfos,
        sizeof(pyRunFinfos) / sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string));
    return &pyRunCinfo;
}

static const Cinfo * pyRunCinfo = PyRun::initCinfo();

PyRun::PyRun()
{
    pystr_ = "";
    globals_ = NULL;
    locals_ = NULL;
}

PyRun::~PyRun()
{
    ;
}

void PyRun::setString(string statement)
{
    pystr_ = statement;
}

string PyRun::getString() const
{
    return pystr_;
}

void PyRun::run(string statement)
{
    PyRun_SimpleString(statement.c_str());
}

void PyRun::process(const Eref & e, ProcPtr p)
{
    cout << "Running: '" << pystr_ << "'" << endl;
    // PyRun_String(pystr_.c_str(), 0, globals_, locals_);
    // PyRun_SimpleString(pystr_.c_str());
       PyEval_EvalCode(compiled_, globals_, locals_);
    
}

/**
   This function does not do anything at this point. It is possible to
   start a separate Python interpreter from here based on a flag. See
   http://www.linuxjournal.com/article/8497 for details.
 */
void PyRun::reinit(const Eref& e, ProcPtr p)
{
    PyObject * main_module;
    if (globals_ == NULL){
        main_module = PyImport_AddModule("__main__");        
        globals_ = PyModule_GetDict(main_module);
    }
    // if (locals_ == NULL){
    //     locals_ = globals_;
    // }
    cout << "Compiling string: " << pystr_ << endl;
    compiled_ = (PyCodeObject*)Py_CompileStringFlags(
        pystr_.c_str(),
        Py_GetProgramName(),
        0,
        Py_file_input);
}


// 
// PyRun.cpp ends here
