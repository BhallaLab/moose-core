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
    static ValueFinfo< PyRun, string > runstring(
        "runString",
        "String to be executed at each time step.",
        &PyRun::setRunString,
        &PyRun::getRunString);

    static ValueFinfo< PyRun, string > initstring(
        "initString",
        "String to be executed at initialization (reinit).",
        &PyRun::setInitString,
        &PyRun::getInitString);

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
        "Runs a specified string. Does not modify existing run or init strings.",
        new OpFunc1< PyRun, string >(&PyRun::run));

    static DestFinfo process(
        "process",
        "Handles process call. Runs the current runString.",
        new ProcOpFunc< PyRun >(&PyRun::process));

    static DestFinfo reinit(
        "reinit",
        "Handles reinit call. Runs the current initString.",
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
        &runstring,
        &initstring,
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

PyRun::PyRun():initstr_(""), runstr_(""),
               globals_(0), locals_(0),
               runcompiled_(0), initcompiled_(0)
{
    ;
}

PyRun::~PyRun()
{
    Py_XDECREF(globals_);
    Py_XDECREF(locals_);
}

void PyRun::setRunString(string statement)
{
    runstr_ = statement;
}

string PyRun::getRunString() const
{
    return runstr_;
}
void PyRun::setInitString(string statement)
{
    initstr_ = statement;
}

string PyRun::getInitString() const
{
    return initstr_;
}

void PyRun::run(string statement)
{
    PyRun_SimpleString(statement.c_str());
}

void PyRun::process(const Eref & e, ProcPtr p)
{
    // PyRun_String(runstr_.c_str(), 0, globals_, locals_);
    // PyRun_SimpleString(runstr_.c_str());
    if (!runcompiled_){
        return;
    }
    PyEval_EvalCode(runcompiled_, globals_, locals_);
    if (PyErr_Occurred()){
        PyErr_Print ();
    } 
}

/**
   This is derived from:
   http://effbot.org/pyfaq/how-do-i-tell-incomplete-input-from-invalid-input.htm
 */
void handleError(bool syntax)
{
    PyObject *exc, *val, *trb;
    char * msg;
    
    if (syntax && PyErr_ExceptionMatches (PyExc_SyntaxError)){           
        PyErr_Fetch (&exc, &val, &trb);        /* clears exception! */
        
        if (PyArg_ParseTuple (val, "sO", &msg, &trb) &&
            !strcmp (msg, "unexpected EOF while parsing")){ /* E_EOF */            
            Py_XDECREF (exc);
            Py_XDECREF (val);
            Py_XDECREF (trb);
        } else {                                  /* some other syntax error */
            PyErr_Restore (exc, val, trb);
            PyErr_Print ();
        }
    } else {                                     /* some non-syntax error */
        PyErr_Print ();
    }                
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
        Py_XINCREF(globals_);
    }
    if (locals_ == NULL){
        locals_ = PyDict_New();
        if (!locals_){
            cerr << "Could not initialize locals dict" << endl;            
        }
    }
    initcompiled_ = (PyCodeObject*)Py_CompileString(
        initstr_.c_str(),
        Py_GetProgramName(),
        Py_file_input);
    if (!initcompiled_){
        cerr << "Error compiling initString" << endl;
        handleError(true);
    } else {
        PyEval_EvalCode(initcompiled_, globals_, locals_);
        if (PyErr_Occurred()){
            PyErr_Print ();
        }
    }
    runcompiled_ = (PyCodeObject*)Py_CompileString(
        runstr_.c_str(),
        Py_GetProgramName(),
        Py_file_input);
    if (!runcompiled_){
        cerr << "Error compiling runString" << endl;
        handleError(true);
    } else {
        PyEval_EvalCode(runcompiled_, globals_, locals_);
        if (PyErr_Occurred()){
            PyErr_Print ();
        }
    }
}


// 
// PyRun.cpp ends here
