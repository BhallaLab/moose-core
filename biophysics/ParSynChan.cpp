/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include "moose.h"
#include <queue>
#include "SynInfo.h"
#include "SynChan.h"
#include "ParSynChan.h"
#include "../element/Neutral.h"

const Cinfo* initParSynChanCinfo()
{
	/** 
	 * This is a shared message to receive Process message from
	 * the scheduler.
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &SynChan::processFunc ) ),
	    new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &SynChan::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) );
	/**
	 * This is a shared message to couple channel to compartment.
	 * The first entry is a MsgSrc to send Gk and Ek to the compartment
	 * The second entry is a MsgDest for Vm from the compartment.
	 */
	static Finfo* channelShared[] =
	{
		new SrcFinfo( "channel", Ftype2< double, double >::global() ),
		new DestFinfo( "Vm", Ftype1< double >::global(), 
				RFCAST( &SynChan::channelFunc ) ),
	};

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* SynChanFinfos[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getGbar ), 
			RFCAST( &SynChan::setGbar )
		),
		new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getEk ), 
			RFCAST( &SynChan::setEk )
		),
		new ValueFinfo( "tau1", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getTau1 ), 
			RFCAST( &SynChan::setTau1 )
		),
		new ValueFinfo( "tau2", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getTau2 ), 
			RFCAST( &SynChan::setTau2 )
		),
		new ValueFinfo( "normalizeWeights", 
			ValueFtype1< bool >::global(),
			GFCAST( &SynChan::getNormalizeWeights ), 
			RFCAST( &SynChan::setNormalizeWeights )
		),
		new ValueFinfo( "Gk", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getGk ), 
			RFCAST( &SynChan::setGk )
		),
		new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &SynChan::getIk ), 
			RFCAST( &SynChan::setIk )
		),

		new ValueFinfo( "numSynapses",
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SynChan::getNumSynapses ), 
			&dummyFunc // Prohibit reassignment of this index.
		),

		new LookupFinfo( "weight",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &SynChan::getWeight ),
			RFCAST( &SynChan::setWeight )
		),

		new LookupFinfo( "delay",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &SynChan::getDelay ),
			RFCAST( &SynChan::setDelay )
		),
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "process", processShared,
			sizeof( processShared ) / sizeof( Finfo* ) ), 
		new SharedFinfo( "channel", channelShared,
			sizeof( channelShared ) / sizeof( Finfo* ) ),

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
		new SrcFinfo( "IkSrc", Ftype1< double >::global() ),
		new SrcFinfo( "origChannel", Ftype2< double, double >::
			global() ),

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		// Arrival of a spike. Arg is time of sending of spike.
		new DestFinfo( "synapse", Ftype1< double >::global(),
				RFCAST( &SynChan::synapseFunc ) ),

		// Sometimes we want to continuously activate the channel
		new DestFinfo( "activation", Ftype1< double >::global(),
				RFCAST( &SynChan::activationFunc ) ),

		// Modulate channel response
		new DestFinfo( "modulator", Ftype1< double >::global(),
				RFCAST( &SynChan::modulatorFunc ) ),
	};

	// SynChan is scheduled after the compartment calculations.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static Cinfo SynChanCinfo(
		"ParSynChan",
		"Mayuresh Kulkarni",
		"Parallel version of SynChan", 
		initNeutralCinfo(),
		SynChanFinfos,
		sizeof( SynChanFinfos )/sizeof(Finfo *),
		ValueFtype1< ParSynChan >::global(),
		schedInfo, 1
	);

	return &SynChanCinfo;
}

static const Cinfo* synChanCinfo = initParSynChanCinfo();

ParSynChan::ParSynChan()
{
}


