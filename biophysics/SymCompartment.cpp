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

// static SrcFinfo2< double, double > *raxial2Out() {
// 	static SrcFinfo2< double, double > raxial2Out( "raxial2Out", 
// 			"Sends out Ra and Vm");
// 	return &raxial2Out;
// }

// static SrcFinfo1< double > *sumRaxial2Out() {
// 	static SrcFinfo1< double> sumRaxial2Out( "sumRaxial2Out", 
// 			"Sends out Ra" );
// 	return &sumRaxial2Out;
// }

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
		"This is a raxial shared message between symmetric compartments.\n"
		"It goes from the tail of the current compartment to the head\n"
		" of the compartment closer to the soma, into another raxial1 message.",
		raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
	);
	static SharedFinfo connecttail( "CONNECTTAIL", 
		"This is a raxial shared message between symmetric compartments."
                                        "It is an alias for raxial1. It goes from the tail of the current\n"
                                        "compartment to the head of the compartment closer to the soma, into\n"
                                        "another raxial1 message.",
                                        raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
	);

	static SharedFinfo connecthead( "CONNECTHEAD", 
                                        "This is a raxial shared message between symmetric compartments.\n"
                                        "It goes from the current compartment to the raxial1 message of \n"
                                        "one further from the soma. The Ra values collected from children and\n"
                                        "sibling nodes are used for computing the equivalent resistance between\n"
                                        "each pair of nodes using star-mesh transformation.\n",
                                        raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
	);

	static SharedFinfo connectcross( "CONNECTCROSS", 
                                         "This is a raxial shared message between symmetric compartments.\n"
                                         "Conceptually, this goes from the tail of the current \n"
                                         "compartment to the tail of a sibling compartment. However,\n"
                                         "this works out to the same as CONNECTHEAD in terms of equivalent\n"
                                         "circuit.  The Ra values collected from siblings and parent node are\n"
                                         "used for computing the equivalent resistance between each pair of\n"
                                         "nodes using star-mesh transformation.\n",
                                         raxial1Shared, sizeof( raxial1Shared ) / sizeof( Finfo* )
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
		// &raxial2,
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
		"Author", "Upi Bhalla; updated and documented by Subhasis Ray",
		"Description", "SymCompartment object, for branching neuron models. In symmetric\n"
                "compartments the axial resistance is equally divided on two sides of\n"
                "the node. The equivalent circuit of the passive compartment becomes:\n"
                "[NOTE: you must use a fixed-width font like Courier for correct rendition of the diagrams below.]\n"
                "                                       \n"
                "         Ra/2    B    Ra/2               \n"                       
                "       A-/\\/\\/\\_____/\\/\\/\\-- C           \n"
                "                 |                      \n"
                "             ____|____                  \n"
                "            |         |                 \n"
                "            |         \\                 \n"
                "            |         / Rm              \n"
                "           ---- Cm    \\                 \n"
                "           ----       /                 \n"
                "            |         |                 \n"
                "            |       _____               \n"
                "            |        ---  Em            \n"
                "            |_________|                 \n"
                "                |                       \n"
                "              __|__                     \n"
                "              /////                     \n" 
                "                                       \n"
                "                                       \n"
                "In case of branching, the B-C part of the parent's axial resistance\n"
                "forms a Y with the A-B part of the children.\n"
                "                               B'              \n"                        
                "                               |               \n"
                "                               /               \n"
                "                               \\              \n"
                "                               /               \n"
                "                               \\              \n"
                "                               /               \n"
                "                               |A'             \n"
                "                B              |               \n"            
                "  A-----/\\/\\/\\-----/\\/\\/\\------|C        \n"
                "                               |               \n"
                "                               |A\"            \n"
                "                               /               \n"
                "                               \\              \n"
                "                               /               \n"
                "                               \\              \n"
                "                               /               \n"
                "                               |               \n"
                "                               B\"             \n"
                "As per basic circuit analysis techniques, the C node is replaced using\n"
                "star-mesh transform. This requires all sibling compartments at a\n"
                "branch point to be connected via CONNECTCROSS messages by the user (or\n"
                "by the cell reader in case of prototypes). For the same reason, the\n"
                "child compartment must be connected to the parent by\n"
                "CONNECTHEAD-CONNECTTAIL message pair. The calculation of the\n"
                "coefficient for computing equivalent resistances in the mesh is done\n"
                "at reinit.",
	};
	static Cinfo symCompartmentCinfo(
			"SymCompartment",
			moose::Compartment::initCinfo(),
			symCompartmentFinfos,
			sizeof( symCompartmentFinfos ) / sizeof( Finfo* ),
			new Dinfo< SymCompartment >(),
                        doc, sizeof(doc)/sizeof(string)
	);

	return &symCompartmentCinfo;
}

static const Cinfo* symCompartmentCinfo = SymCompartment::initCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the SymCompartment class functions.
//////////////////////////////////////////////////////////////////

SymCompartment::SymCompartment():
        coeff_(0.0),
        RaSum_(0.0)
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
	// raxial2Out()->send( e, p->threadIndexInGroup, Ra_, Vm_ ); // to parent and sibs.
}

// Virtual func. Must be called after the 'init' phase.
void SymCompartment::innerReinit( const Eref& e, ProcPtr p )
{
	moose::Compartment::innerReinit( e, p );
        // We don't want to recalculate these every time step - the request... methods are not required
        // requestSumAxial()->send( e, p->threadIndexInGroup );
        // requestSumAxial2()->send( e, p->threadIndexInGroup );
	sumRaxialOut()->send( e, p->threadIndexInGroup, Ra_ );
	// sumRaxial2Out()->send( e, p->threadIndexInGroup, Ra_ );

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

// void SymCompartment::sumRaxial2( double Ra )
// {
//     static int call = 1;
// 	RaSum2_ += Ra_/Ra;
// 	coeff2_ = ( 1 + RaSum2_ ) / 2.0;
// 	// cout << "SymCompartment::sumRaxial " << call << ": coeff2 = " << coeff2_ << "Ra = " << Ra << endl;
//         ++call;
// }

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

// void SymCompartment::raxial2Sym( double Ra, double Vm)
// {
// 	// cout << "SymCompartment " << ":: raxialSym2: Ra = " << Ra << ", Vm = " << Vm << endl;
// 		/*
// 	Ra *= coeff2_;
// 	A_ += Vm / Ra;
// 	B_ += 1.0 / Ra;
// 	Im_ += ( Vm - Vm_ ) / Ra;
// 	*/
//     double R = Ra * coeff2_;
//     // cout << "raxial2Sym:R=" << R << endl;
//     A_ += Vm / R;
//     B_ += 1 / R;
//     Im_ += (Vm - Vm_) / R;
// 	// double invR = 2.0 / ( Ra + Ra_ );
// 	// A_ += Vm * invR;
// 	// B_ += invR;
// 	// Im_ += ( Vm - Vm_ ) * invR;
// }

void SymCompartment::raxialSphere( double Ra, double Vm)
{
	double invR = 2.0 / ( Ra + Ra_ );
	A_ += Vm * invR;
	B_ += invR;
	Im_ += ( Vm - Vm_ ) * invR;
}

/////////////////////////////////////////////////////////////////////

