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

// static SrcFinfo0 *requestSumAxial() {
// 	static SrcFinfo0 requestSumAxial( "requestSumAxial",
// 			"Sends out request for Ra." );
// 	return &requestSumAxial;
// }

static SrcFinfo2< double, double > *raxial2Out() {
	static SrcFinfo2< double, double > raxial2Out( "raxial2Out", 
			"Sends out Ra and Vm");
	return &raxial2Out;
}

static SrcFinfo1< double > *sumRaxial2Out() {
	static SrcFinfo1< double> sumRaxial2Out( "sumRaxial2Out", 
			"Sends out Ra" );
	return &sumRaxial2Out;
}

// static SrcFinfo0 *requestSumAxial2() {
// 	static SrcFinfo0 requestSumAxial2( "requestSumAxial2",
// 			"Sends out request for Ra." );
// 	return &requestSumAxial2;
// }

const Cinfo* SymCompartment::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
    // static DestFinfo process( "process",
    //                   "Handle process call",
    //                   new ProcOpFunc< SymCompartment >( &SymCompartment::process ));
    // static DestFinfo reinit( "reinit",
    //                          "Handles reinit call",
    //                          new ProcOpFunc< SymCompartment >( &Compartment::reinit ));
    // static Finfo * processShared[] =
    // {
    //     &process,
    //     &reinit
    // };
    
    //     static SharedFinfo proc( "proc",
    //     	"This is a shared message to receive Process messages "
    //     	"from the scheduler objects. The Process should be called "
    //     	"_second_ in each clock tick, after the Init message."
    //     	"The first entry in the shared msg is a MsgDest "
    //     	"for the Process operation. It has a single argument, "
    //     	"ProcInfo, which holds lots of information about current "
    //     	"time, thread, dt and so on. The second entry is a MsgDest "
    //     	"for the Reinit operation. It also uses ProcInfo. ",
    //     	processShared, sizeof( processShared ) / sizeof( Finfo* )
    //     );
                                                        
        static DestFinfo raxialSphere( "raxialSphere",
                "Expects Ra and Vm from other compartment. This is a special case when\n"
                "other compartments are evenly distributed on a spherical compartment.",
                new OpFunc2< SymCompartment, double, double >(
                &SymCompartment::raxialSphere)
        );    
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
	// static DestFinfo handleSumRaxialRequest( "handleSumRaxialRequest",
	// 	"Handle request to send back Ra to originating compartment.",
	// 	new EpFunc0< SymCompartment >( 
	// 	&SymCompartment::handleSumRaxialRequest )
	// );

	// The SrcFinfos raxialOut, sumRaxialOut and requestSumAxial
	// are defined above to get them into file-wide scope.

	static Finfo* raxial1Shared[] =
	{
            &raxialSym, &sumRaxial,// &handleSumRaxialRequest, 
            raxialOut(), sumRaxialOut(), //requestSumAxial()
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
	// static DestFinfo handleSumRaxial2Request( "handleSumRaxial2Request",
	// 		"Handles a request to send back Ra to originating compartment.",
	// 		new EpFunc0< SymCompartment >(
	// 			&SymCompartment::handleSumRaxial2Request )
	// );
	// The SrcFinfos raxial2Out, sumRaxial2Out and requestSumAxial2
	// are defined above to get them into file-wide scope.

	static Finfo* raxial2Shared[] =
	{
            &raxial2sym, &sumRaxial2, //&handleSumRaxial2Request,
            raxial2Out(), sumRaxial2Out(), //requestSumAxial2()
		
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

        static Finfo* raxialSphereShared[] = {
            &raxialSphere,
            raxialOut(),
        };
        
        static SharedFinfo connectsphere( "CONNECTSPHERE",
                "This is a shared message between cylindrical and a spherical\n"
                "compartment. It assumes all dendrites are distributed evenly over the\n"
                "soma/sphere. Using CONNECTHEAD/CONNECTTAIL instead connects all\n"
                "dendrites to one point on the soma/sphere.",
                raxialSphereShared, sizeof( raxialSphereShared )/sizeof( Finfo* )
        );

	//////////////////////////////////////////////////////////////////
	static Finfo* symCompartmentFinfos[] = 
	{

	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
	    // The inherited process and init messages do not need to be
		// overridden.
                // &proc,
		&raxial1,
		&connecttail,
		&raxial2,
		&connecthead,
		&connectcross,
                &connectsphere,
                
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

SymCompartment::SymCompartment():
        coeff_(0.0),
        coeff2_(0.0),
        RaSum_(0.0),
        RaSum2_(0.0)
{
	;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////


// void SymCompartment::process( const Eref& e, ProcPtr p )
// {
//     cout << "SymCompartment " << e.id().path() << "::process:  coeff_=" << coeff_ << ", coeff2_=" << coeff2_ << endl;
//     Compartment::process( e, p );
//     // coeff_ = 0.0;
//     // coeff2_ = 0.0;
//     // RaSum_ = 0.0;
//     // RaSum2_ = 0.0;
// }

// Alternates with the 'process' message
void SymCompartment::innerInitProc( const Eref& e, ProcPtr p )
{
	// cout << "SymCompartment " << e.id().path() << ":: innerInitProc: A = " << A_ << ", B = " << B_ << endl;
	raxialOut()->send( e, p->threadIndexInGroup, Ra_, Vm_ ); // to kids
	raxial2Out()->send( e, p->threadIndexInGroup, Ra_, Vm_ ); // to parent and sibs.
}

// Virtual func. Must be called after the 'init' phase.
void SymCompartment::innerReinit( const Eref& e, ProcPtr p )
{
	moose::Compartment::innerReinit( e, p );
        // We don't want to recalculate these every time step - the request... methods are not required
        // requestSumAxial()->send( e, p->threadIndexInGroup );
        // requestSumAxial2()->send( e, p->threadIndexInGroup );
	sumRaxialOut()->send( e, p->threadIndexInGroup, Ra_ );
	sumRaxial2Out()->send( e, p->threadIndexInGroup, Ra_ );

	// cout << "SymCompartment " << e.id().path() << ":: innerReinit: coeff = " << coeff_ << ", coeff2 = " << coeff2_ << endl;
}

// The Compartment and Symcompartment go through an 'init' and then a 'proc'
// during each clock tick. Same sequence applies during reinit.
// This function is called during 'init' phase to send Raxial info around.
void SymCompartment::innerInitReinit( const Eref& e, ProcPtr p )
{
	// cout << "SymCompartment " << e.id().path() << ":: innerInitReinit: coeff = " << coeff_ << ", coeff2 = " << coeff2_ << endl;
	// coeff_ = 0.0;
	// coeff2_ = 0.0;
	// RaSum_ = 0.0;
	// RaSum2_ = 0.0;
	// requestSumAxial()->send( e, p->threadIndexInGroup );
	// requestSumAxial2()->send( e, p->threadIndexInGroup );
}

// void SymCompartment::handleSumRaxialRequest( const Eref& e, const Qinfo* q )
// {
//     cout << "SymCompartment " << e.id().path() << "::handleSumRaxialRequest: Ra_ = " << Ra_ << endl;
// 	sumRaxialOut()->send( e, q->threadNum(), Ra_ );
// }

// void SymCompartment::handleSumRaxial2Request( const Eref& e, const Qinfo* q)
// {
//     cout << "SymCompartment " << e.id().path() << "::handleSumRaxial2Request: Ra_ = " << Ra_ << endl;
// 	sumRaxial2Out()->send( e, q->threadNum(), Ra_ );
// }

void SymCompartment::sumRaxial( double Ra )
{
	RaSum_ += Ra_/Ra;
	coeff_ = ( 1 + RaSum_ ) / 2.0;
	// cout << "SymCompartment::sumRaxial: coeff = " << coeff_ << endl;
}

void SymCompartment::sumRaxial2( double Ra )
{
    static int call = 1;
	RaSum2_ += Ra_/Ra;
	coeff2_ = ( 1 + RaSum2_ ) / 2.0;
	// cout << "SymCompartment::sumRaxial " << call << ": coeff2 = " << coeff2_ << "Ra = " << Ra << endl;
        ++call;
}

void SymCompartment::raxialSym( double Ra, double Vm)
{
	// cout << "SymCompartment " << ":: raxialSym: Ra = " << Ra << ", Vm = " << Vm << endl;
		/*
	Ra *= coeff_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
	*/
    
    double R = Ra * coeff_;
    // cout << "raxialSym:R=" << R << endl;
    A_ += Vm / R;
    B_ += 1.0 / R;
    Im_ += (Vm - Vm_) / R;
	// double invR = 2.0 / ( Ra + Ra_ );
	// A_ += Vm * invR;
	// B_ += invR;
	// Im_ += ( Vm - Vm_ ) * invR;
}

void SymCompartment::raxial2Sym( double Ra, double Vm)
{
	// cout << "SymCompartment " << ":: raxialSym2: Ra = " << Ra << ", Vm = " << Vm << endl;
		/*
	Ra *= coeff2_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
	*/
    double R = Ra * coeff2_;
    // cout << "raxial2Sym:R=" << R << endl;
    A_ += Vm / R;
    B_ += 1 / R;
    Im_ += (Vm - Vm_) / R;
	// double invR = 2.0 / ( Ra + Ra_ );
	// A_ += Vm * invR;
	// B_ += invR;
	// Im_ += ( Vm - Vm_ ) * invR;
}

void SymCompartment::raxialSphere( double Ra, double Vm)
{
	double invR = 2.0 / ( Ra + Ra_ );
	A_ += Vm * invR;
	B_ += invR;
	Im_ += ( Vm - Vm_ ) * invR;
}

/////////////////////////////////////////////////////////////////////

