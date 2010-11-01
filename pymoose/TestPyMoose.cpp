/*******************************************************************
 * File:            TestPyMoose.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
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
using namespace pymoose;

int main(int argc, char **argv)
{
    cout << "Creating context ... " << endl;
    
    PyMooseContext* ctx = PyMooseContext::createPyMooseContext("context", "shell");
    cout << "Successfully created contex ... " << endl;
    cout << "Resetting context .... " << endl;
    
    Compartment c("test");
    HHChannel ch ("Na",c);
    ch.__set_Ek(1.0);
    
    ctx->setClock(0, 1e-4, 0);
    ctx->useClock(PyMooseBase::pathToId("/sched/cj/t0"), "/test");    
    ctx ->reset();
    cout << "Successful reset ... " << endl;
    cout << "Doing step ... " << endl;
    ctx->step(0.05);
    cout << "Successful step ... " << endl;
    Compartment d("test");
    HHChannel dch("Na", d);
    d.connect("channel", &dch, "channel");
    
    vector <string> list = ctx->getMessageList(*dch.__get_id(), false);
    cout << "Outgoing Messages from dch" << endl;
    
    for ( int i = 0; i < list.size(); ++i )
    {
        cout << list[i] << endl;
    }
    
    Neutral root = Neutral("/");
    cout << "Obtained handle for root element: Id " << root.__get_id() << endl;
   
    ctx->deepCopy(*dch.__get_id(), "Na1", *root.__get_id());
    cout << "Deep copy successful" << endl;
    
    vector<Id> children = root.children();
    cout << "Listing children of root..." << endl;
    for ( int i = 0; i < children.size(); ++i ){
	cout << children[i].path() << endl;
    }
    dch.__set_Ek(1.1);

    ctx->readCell( "/soma", "soma.p", 1.0, 1.0, 1.0, 1.0, 1.0 );
    Compartment e("/soma");
    
    cout << "Calling destroy ... " << endl;
    PyMooseContext::destroyPyMooseContext(ctx);
    cout << "Successful destroy ... " << endl;
    return 0;    
}

#endif
