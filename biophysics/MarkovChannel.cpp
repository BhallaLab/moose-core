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
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_errno.h>
#include "MarkovGsl.h"
#include "ChanBase.h"
#include "MarkovChannel.h"

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
			"so on.\n The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	//Field information.
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
	static DestFinfo ligandconc( "ligandconc", 
		"Deals with incoming messages containing information of ligand concentration",
		new OpFunc1< MarkovChannel, double >(&MarkovChannel::handleLigandConc) );

/*	static DestFinfo lookup("lookup",
		"Looks up the rate table corresponding to the (i,j)'th rate i.e. transition from state i to state j.",
		new OpFunc4< MarkovChannel, unsigned int, unsigned int, vector< double >, double >(&MarkovChannel::lookupRate));
*/

	static DestFinfo setoneparam("setoneparam",
		"Sets a one parameter table for the (i,j)'th rate i.e. transition from state"
	  " i to state j. The last parameter is a flag to indicate if the channel is"
		" ligand gated or voltage gated. Note that this table structure should be"
		" used only when the channe l is ligand gated or voltage gated, not both.",
		new OpFunc4< MarkovChannel, vector< unsigned int >, vector< double >, vector< double >, bool >(&MarkovChannel::setOneParamRateTable) );
	
	static DestFinfo settwoparam("settwoparam",	
		"Sets a two parameter table for the (i,j)'th rate i.e. transition from state"
	  "	i to state j. Note that this table structure should be used only when"
		" the channel is ligand gated and voltage gated.",
		new OpFunc3< MarkovChannel, vector< unsigned int >, vector< double >, vector< vector< double > > >(&MarkovChannel::setTwoParamRateTable));

	static DestFinfo setuptables("setuptables",
		"Initializes all rate tables.",
		new OpFunc1< MarkovChannel, unsigned int >(&MarkovChannel::setupRateTables) );

	///////////////////////////////////////////
	static Finfo* MarkovChannelFinfos[] = 
	{
		&numstates,						
		&numopenstates,
		&state,
		&initialstate,
		&labels,
		&gbar,
		&ligandconc,
		&setoneparam,
		&settwoparam,
		&setuptables
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
	numOpenStates_(0),
	rateTables_(0),
	stateForGsl_(0)
//	isInitialized_(false)
{;}

MarkovChannel::MarkovChannel(unsigned int numStates, unsigned int numOpenStates) :
	g_(0), ligandConc_(0), numStates_(numStates), numOpenStates_(numOpenStates)
{
	A_.resize( numStates );
	for ( unsigned int i = 0; i < numStates; ++i )
		A_[i].resize( numStates, 0.0 );

	stateLabels_.resize( numStates );
	state_.resize( numStates );
	initialState_.resize( numStates );
	Gbars_.resize( numOpenStates ) ;
//	isInitialized_ = true;
	stateForGsl_ = new double[numStates];

	rateTables_ = new MarkovRateTable( numStates );

	solver_ = new MarkovGsl();
}

MarkovChannel::~MarkovChannel( )
{	; }

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

void MarkovChannel::setState( vector< double > state ) 
{
	double sumOfProbabilities = 0;

	if ( stateForGsl_ == 0 )
		stateForGsl_ = new double[ state.size() ];

	for ( unsigned int i = 0; i < state.size(); i++)
	{
		stateForGsl_[i] = state[i];
		sumOfProbabilities += state[i];
	}
		
	if ( !doubleEq( sumOfProbabilities, 1.0) ) 
	{
		cerr << "Probabilities of occupation do not sum to 1!\n";
		return;
	}

	state_ = state;	
}

vector< double > MarkovChannel::getInitialState() const 
{
	return initialState_;
}

void MarkovChannel::setInitialState( vector< double > initialState ) 
{
	initialState_ = initialState;
	setState( initialState );
}

vector< double > MarkovChannel::getGbars() const
{
	return Gbars_;
}

void MarkovChannel::setGbars( vector< double > Gbars )
{
	Gbars_ = Gbars;
}

vector< double > MarkovChannel::getOneParamRateTable( unsigned int i, unsigned int j )
{ 
	return rateTables_->getVtChildTable( i, j );
}

void MarkovChannel::setOneParamRateTable( vector< unsigned int > intParams, vector <double > doubleParams, vector< double > table, bool ligandFlag )
{
	rateTables_->setVtChildTable( intParams, doubleParams, table, ligandFlag );
}

vector< vector< double > > MarkovChannel::getTwoParamRateTable( unsigned int i, unsigned int j )
{ 
	return rateTables_->getInt2dChildTable( i, j );
}

void MarkovChannel::setTwoParamRateTable( vector< unsigned int > intParams, vector< double > doubleParams, vector< vector< double > > table )
{
	rateTables_->setInt2dChildTable( intParams, doubleParams, table );	
}

double MarkovChannel::lookupRate( unsigned int i , unsigned int j , vector<double> args )
{
	if ( args.size() == 1 )
		return rateTables_->lookup1D( i, j, args[0] );
	if ( args.size() == 2 )
		return rateTables_->lookup2D( i, j, args[0], args[1] );
	
	cerr << "Error : Either 1 or 2 lookup arguments must be supplied. Returning 0.\n";
	return 0;
}

void MarkovChannel::updateRates()
{
	double x = 0;

	//Rather crude update function. Might be easier to store the variable rates in
	//a separate list and update only those, rather than scan through the entire
	//matrix at every single function evaluation. Profile this later. 
	for ( unsigned int i = 0; i < numStates_; ++i )
	{
		for ( unsigned int j = 0; j < numStates_; ++j )
		{
			//If rate is ligand OR voltage dependent.
			if ( rateTables_->isRateOneParam( i, j ) )
			{
				//Use ligand concentration instead of voltage.
				if ( rateTables_->isRateLigandDep( i, j ) )
					A_[i][j] = rateTables_->lookup1D( i, j, ligandConc_ );
				else
					A_[i][j] = rateTables_->lookup1D( i, j, getVm() );
			}
			
			//If rate is ligand AND voltage dependent. It is assumed that ligand
			//concentration varies along the first dimension.
			if ( rateTables_->isRateTwoParam( i, j ) )
				A_[i][j] = rateTables_->lookup2D( i, j, ligandConc_, getVm() );
		}
	}

	//The values along the diagonal have to be set such that each row sums to
	//zero.
	for ( unsigned int i = 0; i < numStates_; ++i )
	{
		x = 0;
		for ( unsigned int j = 0; j < numStates_; ++j )
		{
			if ( i != j )
				x += A_[i][j];
		}
		A_[i][i] = -x;
	}
}

void MarkovChannel::initConstantRates() 
{
	for (	unsigned int i = 0; i < numStates_; ++i )	
	{
		for ( unsigned int j = 0; j < numStates_; ++j )
		{
			if ( rateTables_->isRateConstant( i, j ) )
				A_[i][j] = rateTables_->lookup1D( i, j, 0.0 );  
			//Doesn't really matter which value is looked up as there is only one
			//entry in the table.
		}
	}
}	

int MarkovChannel::evalGslSystem( double t, const double* state, double* f, void *s)
{
	return static_cast< MarkovChannel* >( s )->innerEvalGslSystem( t, state, f );
}

int MarkovChannel::innerEvalGslSystem( double t, const double* state, double* f )
{
	updateRates();

	//Matrix being accessed along columns, which is a very bad thing in terms of
	//cache optimality. Transposing the matrix during reinit() would be a good idea.
	for ( unsigned int i = 0; i < numStates_; ++i)
	{
		f[i] = 0;
		for ( unsigned int j = 0; j < numStates_; ++j)
			f[i] += state[i] * A_[j][i];
	}

	return GSL_SUCCESS;
}

void MarkovChannel::setupRateTables( unsigned int n )
{
	if ( rateTables_ == 0 )
		rateTables_ = new MarkovRateTable( n );
}

void MarkovChannel::process( const Eref& e, const ProcPtr p ) 
{
	//Get state vector 
	stateForGsl_ = solver_->solve( p->currTime, p->dt, stateForGsl_, numStates_ ); 

	cout << "At time t = " << p->currTime << endl;			

	g_ = 0.0;
	//Cannot quite use the Gbar_ variable of the ChanBase class. The conductance
	//Gk_ calculated here is the "expected conductance" of the channel due to its
	//stochastic nature. 

	for( unsigned int i = 0; i < numOpenStates_; ++i )
	{
		state_[i] = stateForGsl_[i];
		g_ += Gbars_[i] * state_[i];			
	}

	ChanBase::setGk( g_ );
	ChanBase::updateIk();
	ChanBase::process( e, p ); 
}

void MarkovChannel::reinit( const Eref& e, const ProcPtr p )
{
	g_ = 0.0;

	state_ = initialState_;
	for( unsigned int i = 0; i < numStates_; ++i )
		stateForGsl_[i] = state_[i];

	ChanBase::reinit( e, p );	
	initConstantRates();
}

void MarkovChannel::handleLigandConc( double ligandConc )
{
	ligandConc_ = ligandConc;	
}

