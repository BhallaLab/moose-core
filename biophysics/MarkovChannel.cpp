/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "VectorTable.h"
#include "../builtins/Interpol2D.h"
#include "MarkovRateTable.h"
#include "ChanBase.h"
#include "MarkovChannel.h"
#include <gsl/gsl_errno.h>

const Cinfo* MarkovChannel::initCinfo()
{
	//DestFinfos : process and reinit
	static DestFinfo process(	"process",
			"Handles process call",
			new ProcOpFunc< MarkovChannel >( &MarkovChannel::process ) ); 

	static DestFinfo reinit( "reinit", 
			"Handles reinit call",
			new ProcOpFunc< MarkovChannel >( &MarkovChannel::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
			"This is a shared message to receive Process message from the"
			"scheduler. The first entry is a MsgDest for the Process "
			"operation. It has a single argument, ProcInfo, which "
			"holds lots of information about current time, thread, dt and"
			"so on. The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	///////////////////////
	//Field information.
	///////////////////////

	static ValueFinfo< MarkovChannel, unsigned int > numstates( "numstates", 
			"The number of states that the channel can occupy.",
			&MarkovChannel::setNumStates,
			&MarkovChannel::getNumStates 
			);

	static ValueFinfo< MarkovChannel, unsigned int > numopenstates( "numopenstates", 
			"The number of states which are open/conducting.",
			&MarkovChannel::setNumOpenStates,
			&MarkovChannel::getNumOpenStates
			);

	static ValueFinfo< MarkovChannel, vector< string > > labels("labels",
			"Labels for each state.",
			&MarkovChannel::setStateLabels,
			&MarkovChannel::getStateLabels
			);
			
	static ReadOnlyValueFinfo< MarkovChannel, vector< double > > state( "state",
			"This is a row vector that contains the probabilities of finding the channel in each state.",
			&MarkovChannel::getState
			);

	static ValueFinfo< MarkovChannel, vector< double > > initialstate( "initialstate",
			"This is a row vector that contains the probabilities of finding the channel in each state at t = 0. The state of the channel is reset to this value during a call to reinit()",
			&MarkovChannel::setInitialState,
			&MarkovChannel::getInitialState
			);

	static ValueFinfo< MarkovChannel, vector< double > > gbar( "gbar",
			"A row vector containing the conductance associated with each of the open/conducting states.",
			&MarkovChannel::setGbars,
			&MarkovChannel::getGbars
			);

	//MsgDest functions		
	static DestFinfo handleligandconc( "handleligandconc", 
		"Deals with incoming messages containing information of ligand concentration",
		new OpFunc1< MarkovChannel, double >(&MarkovChannel::handleLigandConc) );

	static DestFinfo handlestate("handlestate",
		"Deals with incoming message from MarkovSolver object containing state information of the channel.\n",
		new OpFunc1< MarkovChannel, vector< double > >(&MarkovChannel::handleState) );

	///////////////////////////////////////////
	static Finfo* MarkovChannelFinfos[] = 
	{
		&proc,
		&numstates,						
		&numopenstates,
		&state,
		&initialstate,
		&labels,
		&gbar,
		&handleligandconc,
		&handlestate,
	};

	static string doc[] = 
	{
		"Name", "MarkovChannel",
		"Author", "Vishaka Datta S, 2011, NCBS",
		"Description", "MarkovChannel : Multistate ion channel class." 
		" It deals with ion channels which can be found in one of multiple states,"
	  "	some of which are conducting. This implementation assumes the occurence "
		"of first order kinetics to calculate the probabilities of the channel "
	  "being found in all states. Further, the rates of transition between these "
		"states can be constant, voltage-dependent or ligand dependent (only one "
		"ligand species). The current flow obtained from the channel is calculated " 
		"in a deterministic method by solving the system of differential equations "
	  "obtained from the assumptions above."
	};

	static Cinfo MarkovChannelCinfo(
		"MarkovChannel",
		ChanBase::initCinfo(),
		MarkovChannelFinfos,
		sizeof( MarkovChannelFinfos )/ sizeof( Finfo* ),
		new Dinfo< MarkovChannel >()
		);

	return &MarkovChannelCinfo;
}

static const Cinfo* markovChannelCinfo = MarkovChannel::initCinfo();

MarkovChannel::MarkovChannel() :
	g_(0),
	ligandConc_(0), 
	numStates_(0),
	numOpenStates_(0)
{ ; }
	
MarkovChannel::MarkovChannel(unsigned int numStates, unsigned int numOpenStates) :
	g_(0), ligandConc_(0), numStates_(numStates), numOpenStates_(numOpenStates)
{
	stateLabels_.resize( numStates );
	state_.resize( numStates );
	initialState_.resize( numStates );
	Gbars_.resize( numOpenStates ) ;
}

MarkovChannel::~MarkovChannel( )
{	
	;
}

unsigned int MarkovChannel::getNumStates( ) const
{
	return numStates_;
}

void MarkovChannel::setNumStates( unsigned int numStates ) 
{	
	numStates_ = numStates;
}

unsigned int MarkovChannel::getNumOpenStates( ) const
{
	return numOpenStates_;
}

void MarkovChannel::setNumOpenStates( unsigned int numOpenStates )
{
	numOpenStates_ = numOpenStates;
}

vector< string > MarkovChannel::getStateLabels( ) const
{
	return stateLabels_;
}

void MarkovChannel::setStateLabels( vector< string > stateLabels )
{
	stateLabels_ = stateLabels;
}

vector< double > MarkovChannel::getState ( ) const
{
	return state_;	
}

vector< double > MarkovChannel::getInitialState() const 
{
	return initialState_;
}

void MarkovChannel::setInitialState( vector< double > initialState ) 
{
	initialState_ = initialState;
	state_ = initialState;
}

vector< double > MarkovChannel::getGbars() const
{
	return Gbars_;
}

void MarkovChannel::setGbars( vector< double > Gbars )
{
	Gbars_ = Gbars;
}

/////////////////////////////
//MsgDest functions
////////////////////////////

void MarkovChannel::process( const Eref& e, const ProcPtr p ) 
{
	g_ = 0.0;
	
	//Cannot use the Gbar_ variable of the ChanBase class. The conductance
	//Gk_ calculated here is the "expected conductance" of the channel due to its
	//stochastic nature. 
	for( unsigned int i = 0; i < numOpenStates_; ++i )
		g_ += Gbars_[i] * state_[i];			

	ChanBase::setGk( g_ );
	ChanBase::updateIk();
//	printf("%.15e %.15e %.15e %.15e %.15e\n", state_[0], state_[1], Gbars_[0] * state_[0], Gbars_[1] * state_[1], g_);
	ChanBase::process( e, p ); 
}

void MarkovChannel::reinit( const Eref& e, const ProcPtr p )
{
	g_ = 0.0;

	if ( initialState_.empty() ) 
	{
		cerr << "Initial state has not been set.!\n";
		return;
	}
	state_ = initialState_;

	ChanBase::reinit( e, p );	
}

void MarkovChannel::handleLigandConc( double ligandConc )
{
	ligandConc_ = ligandConc;	
}

void MarkovChannel::handleState( vector< double > state )
{
	state_ = state;
}
