/*******************************************************************
 * File:            TestPyMoose.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-08 04:37:58
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

#ifndef _TESTPYMOOSE_CPP
#define _TESTPYMOOSE_CPP
#include "pymoose.h"

int main(int argc, char **argv)
{
    cout << "Creating context ... " << endl;
    
    PyMooseContext* ctx = PyMooseContext::createPyMooseContext("context", "shell");
    cout << "Successfully created contex ... " << endl;
    cout << "Resetting context .... " << endl;
    Compartment c("test");
    ctx->setClock(0, 1e-4, 0);
    ctx->useClock(PyMooseBase::pathToId("/sched/cj/t0"), "/test");    
    ctx ->reset();
    cout << "Successful reset ... " << endl;
    cout << "Doing step ... " << endl;
    ctx->step(0.05);
    cout << "Successful step ... " << endl;
    cout << "Calling destroy ... " << endl;
    PyMooseContext::destroyPyMooseContext(ctx);
    cout << "Successful destroy ... " << endl;
    return 0;    
}

#endif
