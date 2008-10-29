/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Reaction.h"

extern double getVolScale( Eref e ); // defined in KinCompt.cpp

const Cinfo* initReactionCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &Reaction::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Reaction::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* substrateShared[] =
	{
		new SrcFinfo( "reac", Ftype2< double, double >::global() ),
		new DestFinfo( "sub", Ftype1< double >::global(),
			RFCAST( &Reaction::substrateFunc ) ),
	};
	static Finfo* productShared[] =
	{
		new SrcFinfo( "reac", Ftype2< double, double >::global() ),
		new DestFinfo( "prd", Ftype1< double >::global(),
			RFCAST( &Reaction::productFunc ) ),
	};

	static Finfo* reactionFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "kf", 
			ValueFtype1< double >::global(),
			GFCAST( &Reaction::getRawKf ), 
			RFCAST( &Reaction::setRawKf ) 
		),
		new ValueFinfo( "kb", 
			ValueFtype1< double >::global(),
			GFCAST( &Reaction::getRawKb ), 
			RFCAST( &Reaction::setRawKb ) 
		),

		new ValueFinfo( "Kf", 
			ValueFtype1< double >::global(),
			GFCAST( &Reaction::getKf ), 
			RFCAST( &Reaction::setKf ) 
		),
		new ValueFinfo( "Kb", 
			ValueFtype1< double >::global(),
			GFCAST( &Reaction::getKb ), 
			RFCAST( &Reaction::setKb ) 
		),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "scaleKf", 
			Ftype1< double >::global(),
			RFCAST( &Reaction::scaleKfFunc ) ),
		new DestFinfo( "scaleKb", 
			Ftype1< double >::global(),
			RFCAST( &Reaction::scaleKbFunc ) ),
		new DestFinfo( "rescaleRates", 
			Ftype1< double >::global(),
			RFCAST( &Reaction::rescaleRates ),
			"This handles volume changes of compartment. Argument is ratio of new to old volume." ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "sub", substrateShared,
			sizeof( substrateShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "prd", productShared,
			sizeof( productShared ) / sizeof( Finfo* ) ),
	};

	// Schedule reactions for slower clock, stage 1.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static  Cinfo reactionCinfo(
		"Reaction",
		"Upinder S. Bhalla, 2005, NCBS",
		"Reaction: Reaction class, handles binding and conversion reactions\nnot involving enzymatic steps. Computes reversible reactions\nbut the rates can be set to zero to give irreversibility.\nOrder of substrates and products set by the number of \nmessages between them.",
		initNeutralCinfo(),
		reactionFinfos,
		sizeof(reactionFinfos)/sizeof(Finfo *),
		ValueFtype1< Reaction >::global(),
		schedInfo, 1
	);

	return &reactionCinfo;
}

static const Cinfo* reactionCinfo = initReactionCinfo();

static const Slot substrateSlot =
	initReactionCinfo()->getSlot( "sub.reac" );
static const Slot productSlot =
	initReactionCinfo()->getSlot( "prd.reac" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/**
 * The RawKf and RawKb are rates in units of #/cell. They have the
 * disadvantage that they change when the cell volume is altered, and the
 * units are unfamiliar. They have the advantage that the calculations
 * both for deterministic and stochastic situations, are done in these
 * units. Also they can be assigned without knowing the order of the
 * reaction.
 */

void Reaction::setRawKf( const Conn* c, double value )
{
	static_cast< Reaction* >( c->data() )->kf_ = value;
}

double Reaction::getRawKf( Eref e )
{
	return static_cast< Reaction* >( e.data() )->kf_;
}

void Reaction::setRawKb( const Conn* c, double value )
{
	static_cast< Reaction* >( c->data() )->kb_ = value;
}

double Reaction::getRawKb( Eref e )
{
	return static_cast< Reaction* >( e.data() )->kb_;
}

// Assigns rates in term of regular concentration units, e.g., uM.
// Problem is if the connectivity is set up after the messages, we
// won't know how to scale the rates. kkit sets the raw kf so this issue
// is bypassed.
void Reaction::setKf( const Conn* c, double value )
{
	Eref e = c->target();
	unsigned int numSub = e.e->numTargets( substrateSlot.msg(), e.i );
	if ( numSub <= 1 ) {
		static_cast< Reaction* >( c->data() )->kf_ = value;
	} else {
		double volScale = getVolScale( e );
		volScale = pow( volScale, static_cast< double >( numSub - 1 ) );
		static_cast< Reaction* >( c->data() )->kf_ = value / volScale;
	}
}

double Reaction::getKf( Eref e )
{
	unsigned int numSub = e.e->numTargets( substrateSlot.msg(), e.i );
	if ( numSub <= 1 ) {
		return static_cast< Reaction* >( e.data() )->kf_;
	} else {
		double volScale = getVolScale( e );
		volScale = pow( volScale, static_cast< double >( numSub - 1 ) );
		return static_cast< Reaction* >( e.data() )->kf_ * volScale;
	}
}

void Reaction::setKb( const Conn* c, double value )
{
	Eref e = c->target();
	unsigned int numPrd = e.e->numTargets( productSlot.msg(), e.i );
	if ( numPrd <= 1 ) {
		static_cast< Reaction* >( c->data() )->kb_ = value;
	} else {
		double volScale = getVolScale( e );
		volScale = pow( volScale, static_cast< double >( numPrd - 1 ) );
		static_cast< Reaction* >( c->data() )->kb_ = value / volScale;
	}
}

double Reaction::getKb( Eref e )
{
	unsigned int numPrd = e.e->numTargets( productSlot.msg(), e.i );
	if ( numPrd <= 1 ) {
		return static_cast< Reaction* >( e.data() )->kb_;
	} else {
		double volScale = getVolScale( e );
		volScale = pow( volScale, static_cast< double >( numPrd - 1 ) );
		return static_cast< Reaction* >( e.data() )->kb_ * volScale;
	}
}

/**
 * Ratio is ratio of new vol to old vol.
 * Kf, Kb have units of 1/(conc^(order-1) * sec )
 * new conc = old conc / ratio.
 * so kf = old_kf * ratio^(order-1)
 */
void Reaction::rescaleRates( const Conn* c, double ratio )
{
	Eref e = c->target();
	unsigned int numSub = e.e->numTargets( substrateSlot.msg(), e.i );
	unsigned int numPrd = e.e->numTargets( productSlot.msg(), e.i );
	if ( numSub > 1 ) {
		double scale = pow( ratio, static_cast< double >( numSub - 1 ) );
		static_cast< Reaction* >( c->data() )->kf_ /= scale;
	}
	if ( numPrd > 1 ) {
		double scale = pow( ratio, static_cast< double >( numPrd - 1 ) );
		static_cast< Reaction* >( c->data() )->kb_ /= scale;
	}
}

///////////////////////////////////////////////////
// Shared message function definitions
///////////////////////////////////////////////////

void Reaction::innerProcessFunc( Eref e, ProcInfo info )
{
		send2< double, double >( e, substrateSlot, B_, A_ );
		send2< double, double >( e, productSlot, A_, B_ );
			A_ = kf_;
			B_ = kb_;
}

void Reaction::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< Reaction* >( c->data() )->
		innerProcessFunc( c->target(), p );
}

void Reaction::innerReinitFunc( )
{
		A_ = kf_;
	   	B_ = kb_;
}
		
void Reaction::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< Reaction* >( c->data() )->innerReinitFunc( );
}

void Reaction::substrateFunc( const Conn* c, double n )
{
	static_cast< Reaction* >( c->data() )->A_ *= n;
}

void Reaction::productFunc( const Conn* c, double n )
{
	static_cast< Reaction* >( c->data() )->B_ *= n;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Reaction::scaleKfFunc( const Conn* c, double k )
{
	static_cast< Reaction* >( c->data() )->A_ *= k;
}

void Reaction::scaleKbFunc( const Conn* c, double k )
{
	static_cast< Reaction* >( c->data() )->B_ *= k;
}
