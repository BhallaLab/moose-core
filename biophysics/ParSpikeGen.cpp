/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <math.h>

#include "SpikeGen.h"
#include "ParSpikeGen.h"

const Cinfo* initParSpikeGenCinfo()
{
	/**
	 * This is a shared message to receive Process messages from
	 * the scheduler objects.
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &SpikeGen::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &SpikeGen::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) );
	
	static Finfo* spikeGenFinfos[] = 
	{
		new ValueFinfo( "threshold", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getThreshold ),
			RFCAST( &SpikeGen::setThreshold )
		),
		new ValueFinfo( "refractT", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getRefractT ),
			RFCAST( &SpikeGen::setRefractT )
		),
		new ValueFinfo( "abs_refract", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getRefractT ),
			RFCAST( &SpikeGen::setRefractT )
		),

		/**
		 * The amplitude field of the spikeGen is never used.
		 * \todo: perhaps should deprecate this.
		 */
		new ValueFinfo( "amplitude", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getAmplitude ),
			RFCAST( &SpikeGen::setAmplitude )
		),
		new ValueFinfo( "state", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getState ),
			RFCAST( &SpikeGen::setState )
		),

	//////////////////////////////////////////////////////////////////
	// SharedFinfos
	//////////////////////////////////////////////////////////////////
		process,
		/*
		new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) ),
			*/

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// Sends out a trigger for an event. The time is not 
		// sent - everyone knows the time.
		new SrcFinfo( "event", Ftype1< double >::global() ),

	//////////////////////////////////////////////////////////////////
	// Dest Finfos.
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "Vm", Ftype1< double >::global(),
			RFCAST( &SpikeGen::VmFunc ) ),
	};

	// We want the spikeGen to update after the compartments have done so
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static Cinfo spikeGenCinfo(
				"ParSpikeGen",
				"Mayuresh Kulkarni",
				"Parallel version of SpikeGen",
				initNeutralCinfo(),
				spikeGenFinfos,
				sizeof( spikeGenFinfos ) / sizeof( Finfo* ),
				ValueFtype1< ParSpikeGen >::global(),
				schedInfo, 1
	);

	return &spikeGenCinfo;
}

static const Cinfo* spikeGenCinfo = initParSpikeGenCinfo();

ParSpikeGen::ParSpikeGen()
{
}

