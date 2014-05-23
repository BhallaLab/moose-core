// Leakage.cpp --- 
// 
// Filename: Leakage.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Mon Aug  3 02:32:29 2009 (+0530)
// Version: 
// Last-Updated: Mon Aug 10 11:15:39 2009 (+0530)
//           By: subhasis ray
//     Update #: 71
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: Reimplementation of leakage class of GENESIS
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

#include "header.h"
#include "ChanBase.h"
#include "Leakage.h"

const Cinfo* Leakage::initCinfo()
{
    static DestFinfo process("process",
                             "Handles process call",
                             new ProcOpFunc< Leakage >(&Leakage::process));
    static DestFinfo reinit("reinit", "Handles reinit call",
                            new ProcOpFunc< Leakage >(&Leakage::reinit));
    
    static Finfo* processShared[] = {
        &process,
        &reinit
    };
    
    static SharedFinfo proc("proc", 
                            "This is a shared message to receive Process message from the scheduler. "
                            "The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which "
                            "holds lots of information about current time, thread, dt and so on.\n"
                            "The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo." ,
                            processShared,
                            sizeof(processShared) / sizeof(Finfo*));

    static Finfo * LeakageFinfos[] = {
        &proc,
    };
    
    static string doc[] = {
        "Name", "Leakage",
        "Author", "Subhasis Ray, 2009, NCBS",
        "Description", "Leakage: Passive leakage channel."
    };

    static Dinfo< Leakage > dinfo;
    
    static Cinfo LeakageCinfo(
        "Leakage",
        ChanBase::initCinfo(),
        LeakageFinfos,
        sizeof( LeakageFinfos )/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string ));

    return &LeakageCinfo;
}

static const Cinfo* leakageCinfo = Leakage::initCinfo();


Leakage::Leakage()
{
    ;
}

Leakage::~Leakage()
{
    ;
}
//////// Function definitions

void Leakage::process( const Eref & e, ProcPtr p )
{
    ChanBase::process(e, p);
}

void Leakage::reinit( const Eref & e, ProcPtr p )
{
    ChanBase::reinit(e, p);
}


// 
// Leakage.cpp ends here
