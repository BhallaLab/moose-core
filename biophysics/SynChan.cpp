/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
// #include "SynInfo.h"
#include "Synapse.h"
#include "SynChan.h"

static const double SynE = exp(1.0);

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > IkOut( "IkOut", 
	"Channel current. This message typically goes to concen"
	"objects that keep track of ion concentration." );
static SrcFinfo1< double > permeability( "permeability",
	"Conductance term. Typically goes to GHK object" );
static SrcFinfo2< double, double > channelOut( "channelOut",
	"Sends channel variables Gk and Ek to compartment" );

const Cinfo* SynChan::initCinfo()
{
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< SynChan >( &SynChan::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< SynChan >( &SynChan::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
		"Shared message to receive Process message from scheduler",
		processShared, sizeof( processShared ) / sizeof( Finfo* ) );
		
	/////////////////////////////

	static DestFinfo Vm( "Vm",
			"Handles Vm message coming in from compartment",
			new OpFunc1< SynChan, double >( &SynChan::handleVm ) );

	static Finfo* channelShared[] =
	{
		&channelOut, &Vm
	};

	static  SharedFinfo channel( "channel", 
		"This is a shared message to couple channel to compartment. "
		"The first entry is a MsgSrc to send Gk and Ek to the compartment "
		"The second entry is a MsgDest for Vm from the compartment.",
		channelShared, sizeof( channelShared ) / sizeof( Finfo* )
	);

	static Finfo* ghkShared[] =
	{
		&Vm, &permeability
	};

	static SharedFinfo ghk( "ghk",
		"Message to Goldman-Hodgkin-Katz object",
		ghkShared, sizeof( ghkShared ) / sizeof( Finfo* ) );

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static ValueFinfo< SynChan, double > Gbar( "Gbar",
		"Peak channel conductance.",
        &SynChan::setGbar,
		&SynChan::getGbar
	);

	static ValueFinfo< SynChan, double > Ek( "Ek", 
		"Reversal potential for the synaptic channel.",
        &SynChan::setEk,
		&SynChan::getEk
	);
	static ValueFinfo< SynChan, double > tau1( "tau1", 
		"Decay time constant for the synaptic conductance, tau1 >= tau2.",
        &SynChan::setTau1,
		&SynChan::getTau1
	);
	static ValueFinfo< SynChan, double > tau2( "tau2",
		"Rise time constant for the synaptic conductance, tau1 >= tau2.",
        &SynChan::setTau2,
		&SynChan::getTau2
	);
	static ValueFinfo< SynChan, bool > normalizeWeights( 
		"normalizeWeights", 
		"Flag. If true, the overall conductance is normalized by the "
		"number of individual synapses in this SynChan object.",
        &SynChan::setNormalizeWeights,
		&SynChan::getNormalizeWeights
	);
	static ValueFinfo< SynChan, double > Gk( "Gk", 
		"Conductance of the synaptic channel",
		&SynChan::setGk,
		&SynChan::getGk
	);
	static ReadOnlyValueFinfo< SynChan, double > Ik( "Ik", 
		"Channel current.",
		&SynChan::getIk
	);

	////////////////////////////////////////////////////////////////////
	// FieldElementFinfo definition for Synapses
	////////////////////////////////////////////////////////////////////
	static FieldElementFinfo< SynChan, Synapse > synapse( "synapse",
		"Sets up field Elements for synapse",
		Synapse::initCinfo(),
		&SynChan::getSynapse,
		&SynChan::setNumSynapses,
		&SynChan::getNumSynapses
	);

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo activation( "activation", 
		"Sometimes we want to continuously activate the channel",
		new OpFunc1< SynChan, double >( &SynChan::activation )
	);
	static DestFinfo modulator( "modulator",
		"Modulate channel response",
		new OpFunc1< SynChan, double >( &SynChan::modulator )
	);

	static Finfo* SynChanFinfos[] =
	{
		&proc,			// Shared
		&channel,		// Shared
		&ghk,			// Shared
		&Gbar,			// Value
		&Ek,			// Value
		&tau1,			// Value
		&tau2,			// Value
		&normalizeWeights,	// Value
		&Gk,			// Value
		&Ik,			// ReadOnlyValue
		&activation,	// Dest
		&modulator,	// Dest
		&synapse		// FieldElement
	};

	static string doc[] =
	{
		"Name", "SynChan",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SynChan: Synaptic channel incorporating weight and delay. Does not "
				"handle activity-dependent modification, see HebbSynChan for that. "
				"Very similiar to the old synchan from GENESIS.", 
	};

	static Cinfo SynChanCinfo(
		"SynChan",
		Neutral::initCinfo(),
		SynChanFinfos,
		sizeof( SynChanFinfos )/sizeof(Finfo *),
		new Dinfo< SynChan >()
	);

	return &SynChanCinfo;
}

static const Cinfo* synChanCinfo = SynChan::initCinfo();

SynChan::SynChan()
	: Ek_( 0.0 ), Gk_( 0.0 ), Ik_( 0.0 ), Gbar_( 0.0 ), 
	tau1_( 1.0e-3 ), tau2_( 1.0e-3 ),
	normalizeWeights_( 0 )
{ ; }

SynChan::~SynChan()
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void SynChan::setGbar( double Gbar )
{
	Gbar_ = Gbar;
}
double SynChan::getGbar() const
{
	return Gbar_;
}

void SynChan::setEk( double Ek )
{
	Ek_ = Ek;
}
double SynChan::getEk() const
{
	return Ek_;
}

void SynChan::setTau1( double tau1 )
{
	tau1_ = tau1;
}

double SynChan::getTau1() const
{
	return tau1_;
}

void SynChan::setTau2( double tau2 )
{
    tau2_ = tau2;
}

double SynChan::getTau2() const
{
    return tau2_;
}

void SynChan::setNormalizeWeights( bool value )
{
	normalizeWeights_ = value;
}

bool SynChan::getNormalizeWeights() const
{
	return normalizeWeights_;
}

void SynChan::setGk( double Gk )
{
	Gk_ = Gk;
}
double SynChan::getGk() const
{
	return Gk_;
}

double SynChan::getIk() const
{
	return Ik_;
}

unsigned int SynChan::getNumSynapses() const
{
	return synapses_.size();
}

void SynChan::setNumSynapses( unsigned int i )
{
	; // Illegal operation
}

Synapse* SynChan::getSynapse( unsigned int i )
{
	static Synapse dummy;
	if ( i < synapses_.size() )
		return &( synapses_[i] );
	return &dummy;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SynChan::process( const Eref& e, ProcPtr info )
{
	while ( !pendingEvents_.empty() &&
		pendingEvents_.top().getDelay() <= info->currTime ) {
		activation_ += pendingEvents_.top().getWeight() / info->dt;
		pendingEvents_.pop();
	}
	X_ = modulation_ * activation_ * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	Gk_ = Y_ * norm_;
	Ik_ = ( Ek_ - Vm_ ) * Gk_;
	activation_ = 0.0;
	modulation_ = 1.0;
	channelOut.send( e, info, Gk_, Ek_ );
	IkOut.send( e, info, Ik_ );
	permeability.send( e, info, Gk_ );
}

/*
 * Note that this causes issues if we have variable dt.
 */
void SynChan::reinit( const Eref& e, ProcPtr info )
{
	double dt = info->dt;
	activation_ = 0.0;
	modulation_ = 1.0;
	Gk_ = 0.0;
	Ik_ = 0.0;
	X_ = 0.0;
	Y_ = 0.0;
	xconst1_ = tau1_ * ( 1.0 - exp( -dt / tau1_ ) );
	xconst2_ = exp( -dt / tau1_ );
        if ( doubleEq( tau2_, 0.0 ) ) {
                yconst1_ = 1.0;
                yconst2_ = 0.0;
                norm_ = 1.0;
        } else {
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
                                                ));
                }
        }
        
	// updateNumSynapse( e );
	if ( normalizeWeights_ && synapses_.size() > 0 )
		norm_ /= static_cast< double >( synapses_.size() );
	while ( !pendingEvents_.empty() )
		pendingEvents_.pop();
}

void SynChan::handleVm( double Vm )
{
	Vm_ = Vm;
}

void SynChan::activation( double val )
{
	activation_ += val;
}

void SynChan::modulator( double val )
{
	modulation_ *= val;
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
// #include "SpikeGen.h"

/**
 * Here we set up a SynChan recieving spike inputs from two
 * SpikeGens. The first has a delay of 1 msec, the second of 10.
 * The tau of the SynChan is 1 msec.
 * We test for generation of peak responses at the right time, that
 * is 2 and 11 msec.
 */
 /*
void testSynChan()
{
	cout << "\nTesting SynChan" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* syn = Neutral::create( "SynChan", "syn", n->id(), 
		Id::scratchId() );
	Element* sg1 = Neutral::create( "SpikeGen", "sg1", n->id(), 
		Id::scratchId() );
	Element* sg2 = Neutral::create( "SpikeGen", "sg2", n->id(), 
		Id::scratchId() );
	ASSERT( syn != 0, "creating Synapse" );
	ASSERT( sg1 != 0, "testing Synapse" );
	ASSERT( sg2 != 0, "testing Synapse" );
	ProcInfoBase p;
	p.dt_ = 1.0e-4;
	p.currTime_ = 0;
	SetConn c( syn, 0 );
	SetConn csg1( sg1, 0 );
	SetConn csg2( sg2, 0 );
	bool ret;
	ret = set< double >( syn, "tau1", 1.0e-3 );
	ASSERT( ret, "setup SynChan" );
	ret = set< double >( syn, "tau2", 1.0e-3 );
	ASSERT( ret, "setup SynChan" );
	ret = set< double >( syn, "Gbar", 1.0 );
	ASSERT( ret, "setup SynChan" );

	ret = Eref( sg1 ).add( "event", syn, "synapse" );
	ASSERT( ret, "setup SynChan" );
	ret = Eref( sg2 ).add( "event", syn, "synapse" );

	ASSERT( ret, "setup SynChan" );
	SynChan::reinitFunc( &c, &p );
	
	ret = set< double >( sg1, "threshold", 0.0 );
	ret = set< double >( sg1, "refractT", 1.0 );
	ret = set< double >( sg1, "amplitude", 1.0 );
	ret = set< int >( sg1, "edgeTriggered", 0 );
	ret = set< double >( sg2, "threshold", 0.0 );
	ret = set< double >( sg2, "refractT", 1.0 );
	ret = set< double >( sg2, "amplitude", 1.0 );
	ret = set< int >( sg2, "edgeTriggered", 0 );
	SpikeGen::reinitFunc( &csg1, &p );
	SpikeGen::reinitFunc( &csg2, &p );

	unsigned int temp;
	ret = get< unsigned int >( syn, "numSynapses", temp );
	ASSERT( ret, "setup SynChan" );
	ASSERT( temp == 2, "setup SynChan" );
	ret = lookupSet< double, unsigned int >( syn, "weight", 1.0, 0 );
	ASSERT( ret, "setup SynChan" );
	ret = lookupSet< double, unsigned int >( syn, "delay", 0.001, 0 );
	ASSERT( ret, "setup SynChan" );
	ret = lookupSet< double, unsigned int >( syn, "weight", 1.0, 1 );
	ASSERT( ret, "setup SynChan" );
	ret = lookupSet< double, unsigned int >( syn, "delay", 0.01, 1 );
	ASSERT( ret, "setup SynChan" );

	double dret;
	ret = lookupGet< double, unsigned int >( syn, "weight", dret, 0 );
	ASSERT( dret == 1.0, "setup SynChan" );
	ret = lookupGet< double, unsigned int >( syn, "delay", dret, 0 );
	ASSERT( dret == 0.001, "setup SynChan" );
	ret = lookupGet< double, unsigned int >( syn, "weight", dret, 1 );
	ASSERT( dret == 1.0, "setup SynChan" );
	ret = lookupGet< double, unsigned int >( syn, "delay", dret, 1 );
	ASSERT( dret == 0.01, "setup SynChan" );

	ret = set< double >( sg1, "Vm", 2.0 );
	ret = set< double >( sg2, "Vm", 2.0 );
	ASSERT( ret, "setup SynChan" );
	ret = get< double >( syn, "Gk", dret );
	ASSERT( ret, "setup SynChan" );
	// cout << "dret = " << dret << endl;
	ASSERT( dret == 0.0, "setup SynChan" );
	// Set off the two action potls. They should arrive at 1 and 10 msec
	// respectively
	SpikeGen::processFunc( &csg1, &p );
	SpikeGen::processFunc( &csg2, &p );

	// First 1 msec is the delay, so response should be zero.
	for ( p.currTime_ = 0.0; p.currTime_ < 0.001; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << " dret = " << dret << endl;
	ASSERT( dret == 0.0, "Testing SynChan response" );

	// At 0.5 msec after delay, that is, 1.5 msec, it is at half-tau.
	for ( ; p.currTime_ < 0.0015; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << " dret = " << dret << endl;
	ASSERT( fabs( dret - 0.825  ) < 1e-3 , "Testing SynChan response " );

	// At 1 msec it should be at the peak, almost exactly 1.
	for ( ; p.currTime_ < 0.002; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << " dret = " << dret << endl;
	ASSERT( fabs( dret - 1.0  ) < 2e-3 , "Testing SynChan response" );

	// At 2 msec it is down to 70% of peak.
	for ( ; p.currTime_ < 0.003; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << " dret = " << dret << endl;
	ASSERT( fabs( dret - 0.7  ) < 1e-3 , "Testing SynChan response" );

	// At 3 msec it is down to 38% of peak.
	for ( ; p.currTime_ < 0.004; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << " dret = " << dret << endl;
	ASSERT( fabs( dret - 0.38  ) < 1e-3 , "Testing SynChan response" );

	// Go on to next peak.
	for ( ; p.currTime_ < 0.011; p.currTime_ += p.dt_ )
		SynChan::processFunc( &c, &p );
	ret = get< double >( syn, "Gk", dret );
	// cout << "t = " << p.currTime_ << ", dret = " << dret << endl;
	ASSERT( fabs( dret - 1.0  ) < 2e-3 , "Testing SynChan response 2" );

	////////////////////////////////////////////////////////////////
	// Clear it all up
	////////////////////////////////////////////////////////////////
	set( n, "destroy" );
}
*/
#endif 
