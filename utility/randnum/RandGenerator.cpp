/*******************************************************************
 * File:            RandGenerator.cpp
 * Description:     Interface class for MOOSE to access various
 *                  random number generator.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-03 21:48:17
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

#ifndef _RANDGENERATOR_CPP
#define _RANDGENERATOR_CPP
#include "RandGenerator.h"

const Cinfo * initRandGeneratorCinfo()
{
    static Finfo* processShared[] = 
        {
            new DestFinfo("process", Ftype1< ProcInfo >::global(),
                          RFCAST( &RandGenerator::processFunc )),
            new DestFinfo("reinit", Ftype1<ProcInfo >::global(),
                          RFCAST( &RandGenerator::reinitFunc)),
        };
    static Finfo* process = new SharedFinfo("process", processShared,
                                            sizeof(processShared)/sizeof(Finfo*));
    static Finfo* randGeneratorFinfos[] =
        {
            process,
            new SrcFinfo("output", Ftype1<double>::global()),
        };
    
/*    
    static SchedInfo schedInfo[] = 
        {
            {
                process, 0, 0
            },
        };
*/    
    static Cinfo randGeneratorCinfo("RandGenerator",
                                    "Subhasis Ray",
                                    "Base class for random number generator.",
                                    initNeutralCinfo(),
                                    randGeneratorFinfos,
                                    sizeof(randGeneratorFinfos)/sizeof(Finfo*),
                                    ValueFtype1<RandGenerator>::global());
    return &randGeneratorCinfo;
}

static const unsigned int outputSlot = initRandGeneratorCinfo()->getSlotIndex("output");

RandGenerator::RandGenerator()
{
    rng_ = 0;    
}

RandGenerator::~RandGenerator()
{
    if (rng_)
    {
        delete rng_;
    }    
}

double RandGenerator::getMean(const Element* e)
{
    return static_cast<RandGenerator*>(e->data())->rng_->getMean();
}

double RandGenerator::getVariance(const Element* e)
{
    return static_cast<RandGenerator*>(e->data())->rng_->getVariance();    
}

double RandGenerator::getSample(const Element* e)
{
    return static_cast<RandGenerator*>( e->data())->rng_->getNextSample();
}

void RandGenerator::processFunc( const Conn& c, ProcInfo info )
{
    send1<double>(c.targetElement(), outputSlot, getSample(c.targetElement()));    
}

void RandGenerator::reinitFunc(const Conn& c, ProcInfo info)
{
    cerr << "ERROR: RandGenerator::reinitFunc - this function should never be reached." << endl;
}


#endif
