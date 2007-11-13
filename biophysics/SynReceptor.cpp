/*******************************************************************
 * File:            SynReceptor.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-13 13:27:21
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

#ifndef _SYNRECEPTOR_CPP
#define _SYNRECEPTOR_CPP
#include "moose.h"
#include "SynReceptor.h"
#include <iostream>
using namespace std;

const Cinfo * initSynReceptorCinfo()
{
    static Finfo* processShared[] =
        {
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                           RFCAST( &SynReceptor::processFunc ) ),
            new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                           RFCAST( &SynReceptor::reinitFunc ) ),
	};
    static Finfo* process =  new SharedFinfo( "process", processShared,
                                              sizeof( processShared ) / sizeof( Finfo* ) );
    static Finfo* synReceptorFinfos[] = 
        {
            /// Defines how the postsynaptic potential is related to
            /// the number of quanta released. We are assuming a
            /// linear relation between the two.
            new ValueFinfo( "scale", Ftype1<double>::global(),
                            GFCAST(&SynReceptor::getScale),
                            RFCAST(&SynReceptor::setScale)),
            /// Receive the number of released vesicles.
            new DestFinfo( "releaseCount", Ftype1<double>::global(),
                           RFCAST( &SynReceptor::setReceivedCount)),
            new SrcFinfo( "injectMsg", Ftype1<double>::global()),
            process,
        };
    
    static Cinfo synReceptorCinfo(
        "SynReceptor",
        "Subhasis Ray, NCBS, 2007",
        "SynReceptor: implements a bulk of receptors which receive vesicle contents released by presynaptic terminal and updates the PSP accordingly.",
        initNeutralCinfo(),
        synReceptorFinfos,
        sizeof( synReceptorFinfos )/sizeof(Finfo*),
        ValueFtype1 <SynReceptor>::global());
    return &synReceptorCinfo;
}

static const Cinfo* synReceptorCinfo = initSynReceptorCinfo();
static const unsigned int injectMsgSlot = synReceptorCinfo->getSlotIndex("injectMsg");

double SynReceptor::getScale(const Element * e)
{
    SynReceptor * receptor = static_cast < SynReceptor* >(e->data());
    
    if ( receptor)
    {
        return receptor->scale_;
    }
    else 
    {
        cerr << "ERROR: SynReceptor::getScale - object is null." << endl;
        return 0;        
    }
}

void SynReceptor::setScale(const Conn& c, double scale)
{
    SynReceptor * receptor = static_cast < SynReceptor* >(c.data());
    if ( receptor)
    {
        receptor->scale_ = scale;
    }
    else 
    {
        cerr << "ERROR: SynReceptor::setScale - object is null." << endl;
    }    
}

void SynReceptor::setReceivedCount(const Conn& c, double count)
{
    SynReceptor * receptor = static_cast < SynReceptor* >(c.data());
    if ( receptor)
    {
        receptor->receivedCount_ = count;
    }
    else 
    {
        cerr << "ERROR: SynReceptor::setReceivedCount - object is null." << endl;
    }
}

void SynReceptor::processFunc(const Conn& c, ProcInfo p)
{
    Element* e = c.targetElement();
    SynReceptor* r = NULL;
    
    if ( e )
    {
        r = static_cast<SynReceptor*>(e->data());
    }
    else
    {
        cerr << "ERROR: SynReceptor::process element is NULL." << endl;
        return;
    }
    if (r)
    {
        send1<double>(e, injectMsgSlot, (r->scale_)*(r->receivedCount_));
    }
    else
    {
        cerr << "ERROR: SynReceptor::process SynReceptor object is NULL." << endl;
    }    
}
/**
   TODO: imlement this function
 */
void SynReceptor::reinitFunc(const Conn& c, ProcInfo p)
{
}


#endif
