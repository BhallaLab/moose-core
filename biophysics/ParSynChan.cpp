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
#include <mpi.h>


const Cinfo* initParSynChanCinfo()
{
	/** 
	 * This is a shared message to receive Process message from
	 * the scheduler.
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &ParSynChan::processFunc ) ),
	    	new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &ParSynChan::reinitFunc ) ),
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

		// Accept rank from planarconnect
                new DestFinfo( "recvRank", Ftype1< int >::global(),
                                RFCAST( &ParSynChan::recvRank ) ),

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
static const double SynE = 2.7182818284590452354;
static const int MAX_MPI_PROCESSES = 1024;
static const int SPIKE_TAG = 3;

struct stMPIRecvStatus
{
        MPI_Request request;
        MPI_Status status;
        bool bExecutedRecv;

        stMPIRecvStatus()
        {
                bExecutedRecv = false;
        }

};

ParSynChan::ParSynChan()
{
}

void ParSynChan::recvRank( const Conn& c, int rank )
{
        static_cast< ParSynChan* >( c.data() )->recvRank_.push_back(rank);
}

void ParSynChan::innerProcessFunc( Element* e, ProcInfo info )
{
        static struct stMPIRecvStatus objRecvStatus[MAX_MPI_PROCESSES];
	unsigned int i;
	int iMyRank;
	int flag;
	double tick;

        for (i = 0; i< recvRank_.size(); i++)
        {
                if(objRecvStatus[i].bExecutedRecv == false)
                {
                        MPI_Irecv (&tick, 1, MPI_DOUBLE, recvRank_[i], SPIKE_TAG, MPI_COMM_WORLD, &(objRecvStatus[i].request));
                        objRecvStatus[i].bExecutedRecv = true;
                }

                if(objRecvStatus[i].bExecutedRecv == true)
                {
                        MPI_Test(&objRecvStatus[i].request, &flag, &objRecvStatus[i].status);

                        if(flag == true)
                        {
                                MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
                                cout<<endl<<"Process "<<iMyRank<<" received spike from "<<objRecvStatus[i].status.MPI_SOURCE<<flush;
                                pendingEvents_.push( synapses_[i].event( info->currTime_ ) );
                                objRecvStatus[i].bExecutedRecv = false;
                        }
                }
        }

	SynChan::innerProcessFunc( e, info );

}

unsigned int ParSynChan::updateNumSynapse( const Element* e )
{
        unsigned int n = recvRank_.size();      //synFinfo->numIncoming( e );
        if ( n >= synapses_.size() )
                        synapses_.resize( n );
        return synapses_.size();
}

void ParSynChan::processFunc( const Conn& c, ProcInfo p )
{
        Element* e = c.targetElement();
        static_cast< ParSynChan* >( e->data() )->innerProcessFunc( e, p );
}

/*
 * Note that this causes issues if we have variable dt.
 */
void ParSynChan::innerReinitFunc( Element* e, ProcInfo info )
{
	double dt = info->dt_;
	activation_ = 0.0;
	modulation_ = 1.0;
	Gk_ = 0.0;
	Ik_ = 0.0;
	X_ = 0.0;
	Y_ = 0.0;
	xconst1_ = tau1_ * ( 1.0 - exp( -dt / tau1_ ) );
	xconst2_ = exp( -dt / tau1_ );
	yconst1_ = tau2_ * ( 1.0 - exp( -dt / tau2_ ) );
	yconst2_ = exp( -dt / tau2_ );
	if ( tau1_ == tau2_ ) {
		norm_ = Gbar_ * SynE / tau1_;
	} else {
		double tpeak = tau1_ * tau2_ * log( tau1_ / tau2_ ) / 
			( tau1_ - tau2_ );
		norm_ = Gbar_ * ( tau1_ - tau2_ ) / 
			( tau1_ * tau2_ * ( 
				exp( -tpeak / tau1_ ) - exp( -tpeak / tau2_ )
			) );
	}
	updateNumSynapse( e );
	if ( normalizeWeights_ && synapses_.size() > 0 )
		norm_ /= static_cast< double >( synapses_.size() );
	while ( !pendingEvents_.empty() )
		pendingEvents_.pop();
}

void ParSynChan::reinitFunc( const Conn& c, ProcInfo p )
{
	Element* e = c.targetElement();
	static_cast< ParSynChan* >( e->data() )->innerReinitFunc( e, p );
}

