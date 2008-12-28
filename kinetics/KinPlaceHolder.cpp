/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Neutral.h"
#include "KinPlaceHolder.h"
#include "KinCompt.h"
#include "Reaction.h"
#include "Molecule.h"

const double PI = 3.14159265358979;

// This has to be particularly small because we use SI, where typical
// diffusion rates for proteins are around 1e-12 m^2/sec
const double MIN_DIFFUSION_RATE = 1.0e-20;

/**
 * This class is a placeholder to handle the setup of kinetic models 
 * through readcell or equivalent, where the setup happens on a per-
 * cell-compartment basis similar to a channel. It is necessary to 
 * do this through a placeholder since the SigNeur needs a birds-eye
 * view of the whole cell in order to work out how to decompose the
 * model.
 */

const Cinfo* initKinPlaceHolderCinfo()
{
	static Finfo* kinPlaceHolderFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions. Unusually, all are readonly here.
	///////////////////////////////////////////////////////
		new ValueFinfo( "proto", ValueFtype1< Id >::global(),
			GFCAST( &KinPlaceHolder::getProto ),
			&dummyFunc
		),
		new ValueFinfo( "lambda", ValueFtype1< double >::global(),
			GFCAST( &KinPlaceHolder::getLambda ),
			&dummyFunc
		),
		new ValueFinfo( "method", ValueFtype1< string >::global(),
			GFCAST( &KinPlaceHolder::getMethod ),
			&dummyFunc
		),
		new ValueFinfo( "loadEstimate", ValueFtype1< double >::global(),
			GFCAST( &KinPlaceHolder::getLoadEstimate ),
			&dummyFunc
		),
		new ValueFinfo( "memEstimate", ValueFtype1< unsigned int >::global(),
			GFCAST( &KinPlaceHolder::getMemEstimate ),
			&dummyFunc
		),
		new ValueFinfo( "sigComptLength", ValueFtype1< double >::global(),
			GFCAST( &KinPlaceHolder::getSigComptLength ),
			&dummyFunc
		),
		new ValueFinfo( "numSigCompts", ValueFtype1< unsigned int >::global(),
			GFCAST( &KinPlaceHolder::getNumSigCompts ),
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "build", Ftype0::global(), 
			RFCAST( &KinPlaceHolder::build ),
			"Instantiates the kin model, from the prototype, "
			"including all the required compartments, diffusion, "
			"connections between channels and molecules, and "
			"set up of the solver"
		),
		new DestFinfo( "setup", 
			Ftype3< Id, double, string>::global(), 
			RFCAST( &KinPlaceHolder::setup ),
			"Sets up the info on the PlaceHolder."
			"Talks to the prototype kinetic model to estimate the "
			"load, based on size of prototype model, numerical "
			"method in use, and sometimes the volume of each "
			"compartment. Also estimates memory requirements. "
			"Arguments: Id of prototype kin model, "
			"length constant lambda for diffusion, "
			"string for numerical method to use in calculations."
		),
	};
	
	static string doc[] =
	{
		"Name", "KinPlaceHolder",
		"Author", "Upinder S. Bhalla, 2008, NCBS",
		"Description", 
	"This class is a placeholder to handle the setup of kinetic models "
	"through readcell or equivalent, where the setup happens on a per- "
 	"cell-compartment basis similar to a channel. It is necessary to "
	"do this through a placeholder since the SigNeur needs a birds-eye "
 	"view of the whole cell in order to work out how to decompose the "
 	" model."
	"The placehoder keeps track of the prototype kinetic model and "
	"estimates its CPU and memory use. It also deals with "
	"instantiating the model when all is done."
	};

	static Cinfo kinPlaceHolderCinfo(
	doc,
	sizeof( doc ) / sizeof( string ),
	initNeutralCinfo(),
	kinPlaceHolderFinfos,
	sizeof( kinPlaceHolderFinfos ) / sizeof( Finfo * ),
	ValueFtype1< KinPlaceHolder >::global()
	);

	return &kinPlaceHolderCinfo;
}

static const Cinfo* kinPlaceHolderCinfo = initKinPlaceHolderCinfo();

////////////////////////////////////////////////////////////////////
// Here we set up KinPlaceHolder class functions
////////////////////////////////////////////////////////////////////
KinPlaceHolder::KinPlaceHolder()
	:	
		lambda_( 0.0 ),
		method_( "rk5" ),
		loadEstimate_( 0.0 ),
		memEstimate_( 0 ),
		sigComptLength_( 0.0 ),
		numSigCompts_( 0 )
{ 
	;
}
////////////////////////////////////////////////////////////////////
// Here we set up KinPlaceHolder value fields
////////////////////////////////////////////////////////////////////

Id KinPlaceHolder::getProto( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->proto_;
}

double KinPlaceHolder::getLambda( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->lambda_;
}

string KinPlaceHolder::getMethod( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->method_;
}

double KinPlaceHolder::getLoadEstimate( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->loadEstimate_;
}

unsigned int KinPlaceHolder::getMemEstimate( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->memEstimate_;
}

double KinPlaceHolder::getSigComptLength( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->sigComptLength_;
}

unsigned int KinPlaceHolder::getNumSigCompts( Eref e )
{
	return static_cast< KinPlaceHolder* >( e.data() )->numSigCompts_;
}

////////////////////////////////////////////////////////////////////
// Here we set up KinPlaceHolder Destination functions
////////////////////////////////////////////////////////////////////

/**
 * This is where we should do the estimation etc
 * It needs prior info on the lambda and method to do its stuff.
 */
void KinPlaceHolder::setup( const Conn* c, 
	Id proto, double lambda, string method ) 
{
	static_cast< KinPlaceHolder* >( c->data() )->innerSetup( 
		c->target(), proto, lambda, method );
}

void KinPlaceHolder::build( const Conn* c )
{
	static_cast< KinPlaceHolder* >( c->data() )->innerBuild( 
		c->target() );
}

////////////////////////////////////////////////////////////////////
// Here we set up private KinPlaceHolder class functions.
////////////////////////////////////////////////////////////////////

void KinPlaceHolder::innerSetup( Eref e, 
	Id proto, double lambda, string method ) 
{
	assert( proto.good() );
	assert( lambda > 0.0 );
	proto_ = proto;
	lambda_ = lambda;
	method_ = method;
	// Here we do the estimation stuff.
	set< string >( proto.eref(), "estimateDt", method );
	get< double >( proto.eref(), "loadEstimate", loadEstimate_ );
	get< unsigned int >( proto.eref(), "memEstimate", memEstimate_ );
	Id comptId;
	get< Id >( e, "parent", comptId );
	assert( comptId.good() );
	double electricalComptLength = 0.0;
	get< double >( comptId.eref(), "length", electricalComptLength );
	assert( electricalComptLength > 0.0 );
	numSigCompts_ = 1 + electricalComptLength / lambda;
	sigComptLength_ = electricalComptLength / numSigCompts_;

	double dia = 0.0;
	get< double >( comptId.eref(), "diameter", dia );

	sigComptVolume_ = dia * dia * sigComptLength_ * PI / 4.0;

	loadEstimate_ *= numSigCompts_;
	memEstimate_ *= numSigCompts_;

	cout << "done inner setup for KinPlaceHolder " << 
		e.id().path() << ", numSigCompts = " << numSigCompts_ << endl;
}

void KinPlaceHolder::assignVolumes( Element* dup ) const
{
	assert( dup->cinfo()->isA( initKinComptCinfo() ) );
	for ( unsigned int i = 0; i < numSigCompts_; ++i ) {
		set< double >( Eref( dup, i ), "volume", sigComptVolume_ );
	}
}

void KinPlaceHolder::innerBuild( Eref e ) 
{
	cout << "doing KinPlaceHolder::innerBuild with n = " << numSigCompts_ << endl;
	if ( numSigCompts_ == 0 ) return;
	// Copy or array_copy prototype sig model onto self
	Element* dup;
	if ( numSigCompts_ == 1 ) {
		dup = proto_()->copy( e.e, proto_()->name() );
	} else {
		dup = proto_()->copyIntoArray( 
			e.id(), proto_()->name(), numSigCompts_ );
	}

	assignVolumes( dup ); 

	// Create or array_create diffusion reacs. Need to know if
	// ends will be coupled through fluxes. Assume yes.
	assignDiffusion( dup );

	// Connect up Adaptors
	connectAdaptors( dup );

	// Build solver
	setupSolver( dup );

	// Connect up Solver fluxes. Need to know orientation.
	connectFluxes( dup );
}

void setupSingleDiffusion( Element* mol, double D, double length )
{
	Element* diff = Neutral::create( "Reaction", "diff",
		mol->id(), Id::childId( mol->id() ) );
	assert( diff != 0 );
	Eref( diff ).add( "sub", Eref( mol, 0 ), "reac" );
	Eref( diff ).add( "prd", Eref( mol, 1 ), "reac" );
	double k = D / ( length * length );
	bool ret = set< double >( diff, "kf", k );
	assert( ret );
	ret = set< double >( diff, "kb", k );
	assert( ret );
}

void setupArrayDiffusion( Element* mol, double D, double length )
{
	static const Finfo* kfFinfo = initReactionCinfo()->findFinfo( "kf");
	static const Finfo* kbFinfo = initReactionCinfo()->findFinfo( "kb");
	Element* diff = Neutral::createArray( "Reaction", "diff",
		mol->id(), Id::childId( mol->id() ), mol->numEntries() - 1 );
	assert( diff != 0 );
	length /= mol->numEntries();
	double k = D / ( length * length );
	unsigned int numDiff = diff->numEntries();
	for ( unsigned int i = 0; i < numDiff ; ++i ) {
		Eref e( diff, i );
		e.add( "sub", Eref( mol, i ), "reac" );
		e.add( "prd", Eref( mol, i+1 ), "reac" );
		bool ret = set< double >( e, kfFinfo, k );
		assert( ret );
		ret = set< double >( e, kbFinfo, k );
		assert( ret );
	}
}

/**
 * Scans all children of Element* e
 * If any are pools and have D > 0, create a diffusion object/array.
 */
void KinPlaceHolder::assignDiffusion( Element* mol )
{
	static const Finfo* childListFinfo = 
		initNeutralCinfo()->findFinfo( "childList" );
	static const Finfo* modeFinfo = 
		initMoleculeCinfo()->findFinfo( "mode" );

	vector< Id > kids;
	get< vector< Id > >( mol, childListFinfo, kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i)
		if ( i->index() == 0 )
			assignDiffusion( ( *i )() );
	static const Finfo* dFinfo = initMoleculeCinfo()->findFinfo( "D" );

	if ( mol->cinfo()->isA( initMoleculeCinfo() ) ) {
		double D = 0.0;
		bool ret = get< double >( mol, dFinfo, D );
		assert( ret );
		if ( D < MIN_DIFFUSION_RATE )
			return;
		int mode = 0;
		ret = get< int >( mol, modeFinfo, mode );
		assert( ret );
		if ( mode != 0 ) {
			// Bypass. We only want to handle regular molecules.
			return;
		}

		assert ( mol->numEntries() == numSigCompts_ );
		// Work through different possibilities for diffusion
		if ( mol->numEntries() == 1 )
			return; // If this is a spine, we'll do inter-solver flux.

		if ( mol->numEntries() == 2 )
			setupSingleDiffusion( mol, D, sigComptLength_ );
		else
			setupArrayDiffusion( mol, D, sigComptLength_ );
	}
}

void KinPlaceHolder::connectAdaptors( Element* e ) 
{
}

void KinPlaceHolder::setupSolver( Element* e )
{
}

void KinPlaceHolder::connectFluxes( Element* e )
{
}

