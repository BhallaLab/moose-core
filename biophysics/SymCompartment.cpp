/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

#include "Compartment.h"
#include "SymCompartment.h"

static SrcFinfo2< double, double > *raxialOut() {
	static SrcFinfo2< double, double > raxialOut( "raxialOut", 
			"Sends out Ra and Vm on each timestep" );
	return &raxialOut;
}

static SrcFinfo1< double > *sumRaxialOut() {
	static SrcFinfo1< double > sumRaxialOut( "sumRaxialOut",
		"Sends out Ra" );
	return &sumRaxialOut;
}

static SrcFinfo0 *requestSumAxial() {
	static SrcFinfo0 requestSumAxial( "requestSumAxial",
			"Sends out request for Ra." );
	return &requestSumAxial;
}

static SrcFinfo2< double, double > *raxial2Out() {
	static SrcFinfo2< double, double > raxial2Out( "Raxial2Out", 
			"Sends out Ra and Vm");
	return &raxial2Out;
}

static SrcFinfo1< double > *sumRaxial2Out() {
	static SrcFinfo1< double> sumRaxial2Out( "sumRaxial2Out", 
			"Sends out Ra" );
	return &sumRaxial2Out;
}

static SrcFinfo0 *requestSumAxial2() {
	static SrcFinfo0 requestSumAxial2( "requestSumAxial2",
			"Sends out request for Ra." );
	return &requestSumAxial2;
}

const Cinfo* SymCompartment::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	static DestFinfo raxialSym( "raxialSym", 
		"Expects Ra and Vm from other compartment.",
		new OpFunc2< SymCompartment, double, double >( 
			&SymCompartment::raxialSym )
	);
	static DestFinfo sumRaxial( "sumRaxial", 
		"Expects Ra from other compartment.",
		new OpFunc1< SymCompartment, double >( 
		&SymCompartment::sumRaxial )
	);
	static DestFinfo handleSumRaxialRequest( "handleSumRaxialRequest",
		"Handle request to send back Ra to originating compartment.",
		new EpFunc0< SymCompartment >( 
		&SymCompartment::handleSumRaxialRequest )
	);

	// The SrcFinfos raxialOut, sumRaxialOut and requestSumAxial
	// are defined above to get them into file-wide scope.

	static Finfo* raxial1Shared[] =
	{
		&raxialSym, &sumRaxial, &handleSumRaxialRequest, 
		raxialOut(), sumRaxialOut(), requestSumAxial()
	};

	static SharedFinfo raxial1( "raxial1",
		"This is a raxial shared message between symmetric compartments."
		"It goes from the tail of the current compartment to the head "
		" of the compartment closer to the soma, into an raxial2 message.",
		raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
	);
	static SharedFinfo connecttail( "CONNECTTAIL", 
		"This is a raxial shared message between symmetric compartments."
		"It is an alias for raxial1.",
		raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
	);
	/////////////////////////////////////////////////////////////////////

	static DestFinfo raxial2sym( "raxial2sym", 
			"Expects Ra and Vm from other compartment.",
			new OpFunc2< SymCompartment, double, double >( 
			&SymCompartment::raxial2Sym )
	);
	static DestFinfo sumRaxial2( "sumRaxial2", 
			"Expects Ra from other compartment.",
			new OpFunc1< SymCompartment, double >( 
			&SymCompartment::sumRaxial2 )
	);
	static DestFinfo handleSumRaxial2Request( "handleSumRaxial2Request",
			"Handles a request to send back Ra to originating compartment.",
			new EpFunc0< SymCompartment >(
				&SymCompartment::handleSumRaxial2Request )
	);
	// The SrcFinfos raxial2Out, sumRaxial2Out and requestSumAxial2
	// are defined above to get them into file-wide scope.

	static Finfo* raxial2Shared[] =
	{
		&raxial2sym, &sumRaxial2, &handleSumRaxial2Request,
		raxial2Out(), sumRaxial2Out(), requestSumAxial2()
		
	};

	static SharedFinfo raxial2( "raxial2", 
		"This is a raxial2 shared message between symmetric compartments."
		"It goes from the head of the current compartment to "
		"the raxial1 message of a compartment further away from the soma",
		raxial2Shared, sizeof( raxial2Shared ) / sizeof( Finfo* )
	);

	static SharedFinfo connecthead( "CONNECTHEAD", 
		"This is a raxial2 shared message between symmetric compartments."
		"It is an alias for raxial2."
		"It goes from the current compartment to the raxial1 message of "
		"one further from the soma",
		raxial2Shared, sizeof( raxial2Shared ) / sizeof( Finfo* )
	);

	static SharedFinfo connectcross( "CONNECTCROSS", 
		"This is a raxial2 shared message between symmetric compartments."
		"It is an alias for raxial2."
		"Conceptually, this goes from the tail of the current "
		"compartment to the tail of a sibling compartment. However,"
		"this works out to the same as CONNECTHEAD in terms of equivalent"
		"circuit.",
		raxial2Shared, sizeof( raxial2Shared ) / sizeof( Finfo* )
	);

	//////////////////////////////////////////////////////////////////
	static Finfo* symCompartmentFinfos[] = 
	{

	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
	    // The inherited process and init messages do not need to be
		// overridden.
		&raxial1,
		&connecttail,
		&raxial2,
		&connecthead,
		&connectcross

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
	};

	// static SchedInfo schedInfo[] = { { process, 0, 0 }, { init, 0, 1 } };
	
	static string doc[] =
	{
		"Name", "SymCompartment",
		"Author", "Upi Bhalla",
		"Description", "SymCompartment object, for branching neuron models.",
	};
	static Cinfo symCompartmentCinfo(
			"SymCompartment",
			moose::Compartment::initCinfo(),
			symCompartmentFinfos,
			sizeof( symCompartmentFinfos ) / sizeof( Finfo* ),
			new Dinfo< SymCompartment >()
	);

	return &symCompartmentCinfo;
}

static const Cinfo* symCompartmentCinfo = SymCompartment::initCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the SymCompartment class functions.
//////////////////////////////////////////////////////////////////

SymCompartment::SymCompartment()
{
	;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

/*
void SymCompartment::innerProcessFunc( Element* e, ProcInfo p )
{
	A_ += Inject_ + sumInject_ + Em_ * invRm_; 
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * p->dt_ / Cm_ );
		Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
	} else {
		Vm_ += ( A_ - Vm_ * B_ ) * p->dt_ / Cm_;
	}
	A_ = 0.0;
	B_ = invRm_; 
	Im_ = 0.0;
	sumInject_ = 0.0;
	// Send out the channel messages
	send1< double >( e, channelSlot, Vm_ );
	// Send out the message to any SpikeGens.
	send1< double >( e, VmSlot, Vm_ );
	// Send out the axial messages
	// send1< double >( e, axialSlot, Vm_ );
	// Send out the raxial messages
	// send2< double >( e, raxialSlot, Ra_, Vm_ );
}
*/

// Alternates with the 'process' message
void SymCompartment::innerInitProc( const Eref& e, ProcPtr p )
{
	raxialOut()->send( e, p->threadIndexInGroup, Ra_, Vm_ ); // to kids
	raxial2Out()->send( e, p->threadIndexInGroup, Ra_, Vm_ ); // to parent and sibs.
}

// Virtual func. Must be called after the 'init' phase.
void SymCompartment::innerReinit( const Eref& e, ProcPtr p )
{
	moose::Compartment::innerReinit( e, p );

	coeff_ *= Ra_;
	coeff_ = ( 1 + coeff_ ) / 2.0;

	coeff2_ *= Ra_;
	coeff2_ = ( 1 + coeff2_ ) / 2.0;
}

// The Compartment and Symcompartment go through an 'init' and then a 'proc'
// during each clock tick. Same sequence applies during reinit.
// This funciton is called during 'init' phase to send Raxial info around.
void SymCompartment::innerInitReinit( const Eref& e, ProcPtr p )
{
	coeff_ = 0.0;
	coeff2_ = 0.0;
	requestSumAxial()->send( e, p->threadIndexInGroup );
	requestSumAxial2()->send( e, p->threadIndexInGroup );
}

void SymCompartment::handleSumRaxialRequest( const Eref& e, const Qinfo* q )
{
	sumRaxialOut()->send( e, q->threadNum(), Ra_ );
}

void SymCompartment::handleSumRaxial2Request( const Eref& e, const Qinfo* q)
{
	sumRaxial2Out()->send( e, q->threadNum(), Ra_ );
}

void SymCompartment::sumRaxial( double Ra )
{
	coeff_ += 1.0 / Ra;
}

void SymCompartment::sumRaxial2( double Ra )
{
	coeff2_ += 1.0 / Ra;
}

void SymCompartment::raxialSym( double Ra, double Vm)
{
	Ra *= coeff_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void SymCompartment::raxial2Sym( double Ra, double Vm)
{
	Ra *= coeff2_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

/////////////////////////////////////////////////////////////////////

