/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "SigNeur.h"
#include "../shell/Shell.h"
#include "../element/Wildcard.h"

static const double PI = 3.1415926535;

const Cinfo* initSigNeurCinfo()
{
	static Finfo* sigNeurFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////

		new ValueFinfo( "sigDt", 
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getSigDt ), 
			RFCAST( &SigNeur::setSigDt )
		),
		new ValueFinfo( "cellDt", 
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getCellDt ), 
			RFCAST( &SigNeur::setCellDt )
		),

		new ValueFinfo( "Dscale", 
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getDscale ), 
			RFCAST( &SigNeur::setDscale )
		),
		new ValueFinfo( "parallelMode", 
			ValueFtype1< int >::global(),
			GFCAST( &SigNeur::getParallelMode ), 
			RFCAST( &SigNeur::setParallelMode )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "build", Ftype0::global(),
			RFCAST( &SigNeur::build )
		),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	};

	// Schedule it to tick 1 stage 0
	// static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "SigNeur",
		"Author", "Upinder S. Bhalla, 2008-2009, NCBS",
		"Description", 
		"SigNeur: Multiscale simulation setup object for doing "
		"combined electrophysiological and signaling models of "
		"neurons. Takes the geometry from the neuronal model and "
		"sets up diffusion between signaling models to fit in this "
		"geometry. Assumes that the extended ReadCell has loaded in "
		"the KinPlaceHolder and adaptor objects to specify how to do "
		"the sig/neur interface."
	};

	static Cinfo sigNeurCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		sigNeurFinfos,
		sizeof( sigNeurFinfos )/sizeof(Finfo *),
		ValueFtype1< SigNeur >::global()
	);

	// methodMap.size(); // dummy function to keep compiler happy.

	return &sigNeurCinfo;
}

static const Cinfo* sigNeurCinfo = initSigNeurCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SigNeur::SigNeur()
	: 	
		sigDt_( 10.0e-3 ),
		cellDt_( 50.0e-6 ),
		Dscale_( 1.0 ),
		parallelMode_( 0 )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void SigNeur::setSigDt( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->sigDt_ = value;
}

double SigNeur::getSigDt( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->sigDt_;
}


void SigNeur::setCellDt( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->cellDt_ = value;
}

double SigNeur::getCellDt( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cellDt_;
}


void SigNeur::setDscale( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->Dscale_ = value;
}

double SigNeur::getDscale( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->Dscale_;
}

void SigNeur::setParallelMode( const Conn* c, int value )
{
	static_cast< SigNeur* >( c->data() )->parallelMode_ = value;
}

int SigNeur::getParallelMode( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->parallelMode_;
}

void SigNeur::build( const Conn* c )
{
	static_cast< SigNeur* >( c->data() )->innerBuild( c );
}

//////////////////////////////////////////////////////////////////
// Here we set up some of the messier inner functions.
//////////////////////////////////////////////////////////////////

void SigNeur::innerBuild( const Conn* c )
{
	vector< Id > sigs;
	// read in list of sigs here
	allChildren( c->target().id(), "ISA=KinPlaceHolder", 0, sigs );
	cout << "SigNeur::innerBuild found " << sigs.size() << 
		" sigs " << endl;
	for ( vector< Id >::iterator i = sigs.begin(); i != sigs.end(); ++i)
		set( i->eref(), "build" );
	schedule( c->target() );
}


bool SigNeur::traverseCell( Eref me )
{
	return 1;
}

void SigNeur::schedule( Eref me )
{
/*
	static const Finfo* lookupChildFinfo =
		initNeutralCinfo()->findFinfo( "lookupChild" );
	Id kinId;
	lookupGet< Id, string >( me, lookupChildFinfo, kinId, "kinetics" );
	assert( kinId.good() );

	Id cellId;
	lookupGet< Id, string >( me, lookupChildFinfo, cellId, "cell" );
	assert( cellId.good() );

	SetConn c( Id::shellId().eref() );
	Shell::setClock( &c, 0, cellDt_, 0 );
	Shell::setClock( &c, 1, cellDt_, 1 );
	Shell::setClock( &c, 2, sigDt_, 0 );
	Shell::setClock( &c, 3, sigDt_, 1 );

	set< string >( cellId.eref(), "method", cellMethod_ );
	set< string >( kinId.eref(), "method", dendMethod_ );
	if ( separateSpineSolvers_ ) {
		vector< Id > kids;
		get< vector< Id > >( spine_.eref(), "childList", kids );
		cout << "Setting separate spine method " << spineMethod_ <<
			" to " << kids.size() << " spines\n";
		for ( vector< Id >::iterator i = kids.begin(); 
			i != kids.end(); ++i )
			set< string >( i->eref(), "method", spineMethod_ );
	}

	Shell::useClock( &c, "t2", "/sig/kinetics", "process" );
	Shell::useClock( &c, "t2", "/sig/kinetics/solve/hub", "process" );
	Shell::useClock( &c, "t2", "/sig/kinetics/solve/integ", "process" );

	Shell::useClock( &c, "t3", "/sig/cell/##[][TYPE==Adaptor]", "process" );
	*/
}
