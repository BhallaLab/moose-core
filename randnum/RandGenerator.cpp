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
            new ValueFinfo("sample", ValueFtype1<double>::global(),
                           GFCAST( &RandGenerator::getSample),
                           &dummyFunc),
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &RandGenerator::getMean),
                           &dummyFunc),
            new ValueFinfo("variance", ValueFtype1<double>::global(),
                           GFCAST( &RandGenerator::getVariance),
                           &dummyFunc),

            process,
            new SrcFinfo("output", Ftype1<double>::global()),
        };

    static SchedInfo schedInfo[] = 
        {
            {
                process, 0, 0 
            }
        };
    
    static string doc[] =
	{
		"Name", "RandGenerator",
		"Author", "Subhasis Ray",
		"Description", "Base class for random number generator.",
	};
    static Cinfo randGeneratorCinfo(
                                    doc,
				    sizeof( doc ) / sizeof( string ),
                                    initNeutralCinfo(),
                                    randGeneratorFinfos,
                                    sizeof(randGeneratorFinfos)/sizeof(Finfo*),
                                    ValueFtype1<RandGenerator>::global(),
                                    schedInfo, 1);
    return &randGeneratorCinfo;
}

static const Slot outputSlot = initRandGeneratorCinfo()->getSlot("output");

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

double RandGenerator::getMean(Eref e)
{
    Probability* gen = static_cast<RandGenerator*>(e.data())->rng_;
    
    if (gen)
    {
        return gen->getMean();
    }
    else
    {
        cerr << "WARNING: RandGenerator::getMean - parameters not set for object " << e.e->name() << endl;
        return 0;
    }            
}

double RandGenerator::getVariance(Eref e)
{
    Probability* gen = static_cast<RandGenerator*>(e.data())->rng_;
    if (gen)
    {
        return gen->getVariance();    
    }
    else
    {
        cerr << "WARNING: RandGenerator::getVariance - parameters not set for object " << e.e->name() << endl;
        return 0;
    }
    
        
}

double RandGenerator::getSample(Eref e)
{
    Probability* gen = static_cast<RandGenerator*>(e.data())->rng_;
    if (gen)
    {
        return gen->getNextSample();
    }
    else
    {
        cerr << "WARNING: RandGenerator::getSample  - parameters not set for object " << e.e->name() << endl;
        return 0;
    }    
}

void RandGenerator::processFunc( const Conn* c, ProcInfo info )
{
    send1<double>(c->target(), outputSlot, getSample(c->target()));    
}

void RandGenerator::reinitFunc(const Conn* c, ProcInfo info)
{
    RandGenerator* generator = static_cast < RandGenerator* >(c->data());
    generator->innerReinitFunc(c, info);    
}

void RandGenerator::innerReinitFunc(const Conn* c, ProcInfo info)
{
    cerr << "RandGenerator::innerReinitFunc() - this function should never be reached. Guilty party: " << c->target().e->name() << endl;
}

#endif
