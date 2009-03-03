/*******************************************************************
 * File:            UniformRng.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-01 11:30:20
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

#ifndef _UNIFORMRNG_CPP
#define _UNIFORMRNG_CPP
#include "basecode/moose.h"
#include "randnum.h"
#include "UniformRng.h"
#include "Uniform.h"

extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initUniformRngCinfo()
{
    static Finfo* uniformRngFinfos[] =
        {
            new ValueFinfo("min", ValueFtype1 <double>::global(),
                           GFCAST( &UniformRng::getMin),
                           RFCAST( &UniformRng::setMin),
						   "The lower bound on the numbers generated " ),
            new ValueFinfo("max", ValueFtype1 <double>::global(),
                           GFCAST( &UniformRng::getMax),
                           RFCAST( &UniformRng::setMax),
						   "The upper bound on the numbers generated" ),
                           
        };
    
    
    static string doc[] =
	{
		"Name", "UniformRng",
		"Author", "Subhasis Ray",
		"Description", "Uniformly distributed random number generator.",
	};
    static Cinfo uniformRngCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),
                                initRandGeneratorCinfo(),
                                uniformRngFinfos,
                                sizeof(uniformRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<UniformRng>::global()
                                );
    return &uniformRngCinfo;
}

static const Cinfo* uniformRngCinfo = initUniformRngCinfo();

double UniformRng::getMin(Eref e)
{
    return static_cast <Uniform *> (static_cast<UniformRng*> (e.data())->rng_)->getMin();
}

double UniformRng::getMax(Eref e)
{
    return static_cast <Uniform *> (static_cast<UniformRng*> (e.data())->rng_)->getMax();
}

void UniformRng::setMin(const Conn* c, double min)
{
    UniformRng* obj = static_cast <UniformRng*> (c->data());
    if (obj)
    {
        if (obj->rng_)
        {
            static_cast<Uniform *> (obj->rng_)->setMin(min);
        }
        else 
        {
            cerr << "UniformRng::setMin() - generator Uniform object is null." << endl;
        }        
    }
    else
    {
        cerr << "UniformRng::setMin() - connection target is NULL." << endl;
    }
}
void UniformRng::setMax(const Conn* c, double max)
{
    UniformRng* obj = static_cast <UniformRng*> (c->data());
    if (obj)
    {
        if (obj->rng_)
        {
            static_cast<Uniform *> (obj->rng_)->setMax( max );
        }
        else 
        {
            cerr << "UniformRng::setMax() - generator Uniform object is null." << endl;
        }        
    }
    else
    {
        cerr << "UniformRng::setMax() - connection target is NULL." << endl;
    }
}


UniformRng::UniformRng():RandGenerator()
{
    rng_ = new Uniform();
}

void UniformRng::innerReinitFunc(const Conn* conn, ProcInfo info)
{
    ;    /* no use */
}

#ifdef DO_UNIT_TESTS
void testUniformRng()
{
    cout << "testUniformRng(): yet to be implemented" << endl;
}

#endif
#endif
