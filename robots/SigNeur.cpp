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
		new ValueFinfo( "cellProto", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getCellProto ), 
			RFCAST( &SigNeur::setCellProto ) 
		),
		new ValueFinfo( "spineProto", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSpineProto ), 
			RFCAST( &SigNeur::setSpineProto )
		),
		new ValueFinfo( "dendProto", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getDendProto ), 
			RFCAST( &SigNeur::setDendProto )
		),
		new ValueFinfo( "somaProto", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSomaProto ), 
			RFCAST( &SigNeur::setSomaProto )
		),

		new ValueFinfo( "cell", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getCell ), 
			&dummyFunc
		),
		new ValueFinfo( "spine", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSpine ), 
			&dummyFunc
		),
		new ValueFinfo( "dend", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getDend ), 
			&dummyFunc
		),
		new ValueFinfo( "soma", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSoma ), 
			&dummyFunc
		),

		new ValueFinfo( "cellMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getCellMethod ), 
			RFCAST( &SigNeur::setCellMethod )
		),
		new ValueFinfo( "spineMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getSpineMethod ), 
			RFCAST( &SigNeur::setSpineMethod )
		),
		new ValueFinfo( "dendMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getDendMethod ), 
			RFCAST( &SigNeur::setDendMethod )
		),
		new ValueFinfo( "somaMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getSomaMethod ), 
			RFCAST( &SigNeur::setSomaMethod )
		),

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
		new ValueFinfo( "lambda", 
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getLambda ), 
			RFCAST( &SigNeur::setLambda )
		),
		new ValueFinfo( "parallelMode", 
			ValueFtype1< int >::global(),
			GFCAST( &SigNeur::getParallelMode ), 
			RFCAST( &SigNeur::setParallelMode )
		),
		new ValueFinfo( "updateStep", // Time between sig<->neuro updates
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getUpdateStep ), 
			RFCAST( &SigNeur::setUpdateStep )
		),
		new LookupFinfo( "channelMap", // Mapping from channels to sig mols
			LookupFtype< string, string >::global(),
			GFCAST( &SigNeur::getChannelMap ), 
			RFCAST( &SigNeur::setChannelMap )
		),
		new LookupFinfo( "calciumMap",  // Mapping from calcium to sig.
			LookupFtype< string, string >::global(),
			GFCAST( &SigNeur::getCalciumMap ), 
			RFCAST( &SigNeur::setCalciumMap )
		),
		new ValueFinfo( "calciumScale",
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getCalciumScale ), 
			RFCAST( &SigNeur::setCalciumScale )
		),
		new ValueFinfo( "dendInclude",
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getDendInclude ), 
			RFCAST( &SigNeur::setDendInclude )
		),
		new ValueFinfo( "dendExclude",
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getDendExclude ), 
			RFCAST( &SigNeur::setDendExclude )
		),
	// Would be nice to have a way to include synaptic input into
	// the mGluR input.
	
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
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SigNeur: Multiscale simulation setup object for doing combined electrophysiological "
				"and signaling models of neurons. Takes the geometry from the neuronal model and "
				"sets up diffusion between signaling models to fit in this geometry. Arranges "
				"interfaces between channel conductances and molecular species representing "
				"channels.Also interfaces calcium conc in the two kinds of model.",
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
		numSpine_( 0 ), 
		numNeck_( 0 ), 
		numDend_( 0 ), 
		numSoma_( 0 ), 
		cellMethod_( "hsolve" ), 
		spineMethod_( "rk5" ), 
		dendMethod_( "rk5" ), 
		somaMethod_( "rk5" ), 
		sigDt_( 10.0e-3 ),
		cellDt_( 50.0e-6 ),
		Dscale_( 1.0 ),
		lambda_( 10.0e-6 ),
		parallelMode_( 0 ),
		updateStep_( 1.0 ),
		calciumScale_( 1.0 ),
		dendInclude_( "" ),
		dendExclude_( "" )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

// prototypes
void SigNeur::setCellProto( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->cellProto_ = value;
}

Id SigNeur::getCellProto( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cellProto_;
}

void SigNeur::setSpineProto( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->spineProto_ = value;
}

Id SigNeur::getSpineProto( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->spineProto_;
}

void SigNeur::setDendProto( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->dendProto_ = value;
}

Id SigNeur::getDendProto( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dendProto_;
}

void SigNeur::setSomaProto( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->somaProto_ = value;
}

Id SigNeur::getSomaProto( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->somaProto_;
}

// created arrays
Id SigNeur::getCell( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cell_;
}

Id SigNeur::getSpine( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->spine_;
}

Id SigNeur::getDend( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dend_;
}

Id SigNeur::getSoma( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->soma_;
}


void SigNeur::setCellMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->cellMethod_ = value;
}

string SigNeur::getCellMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cellMethod_;
}

void SigNeur::setSpineMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->spineMethod_ = value;
}

string SigNeur::getSpineMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->spineMethod_;
}

void SigNeur::setDendMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->dendMethod_ = value;
}

string SigNeur::getDendMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dendMethod_;
}

void SigNeur::setSomaMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->somaMethod_ = value;
}

string SigNeur::getSomaMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->somaMethod_;
}

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

void SigNeur::setLambda( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->lambda_ = value;
}

double SigNeur::getLambda( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->lambda_;
}

void SigNeur::setParallelMode( const Conn* c, int value )
{
	static_cast< SigNeur* >( c->data() )->parallelMode_ = value;
}

int SigNeur::getParallelMode( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->parallelMode_;
}

void SigNeur::setUpdateStep( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->updateStep_ = value;
}

double SigNeur::getUpdateStep( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->updateStep_;
}

void SigNeur::setCalciumMap( const Conn* c, string val, const string& i )
{
	static_cast< SigNeur* >( c->data() )->calciumMap_[ i ] = val;
}

string SigNeur::getCalciumMap( Eref e, const string& i )
{
	SigNeur* sn = static_cast< SigNeur* >( e.data() );
	map< string, string >::iterator j = sn->calciumMap_.find( i );
	if ( j != sn->calciumMap_.end() )
		return j->second;
	return "";
}

void SigNeur::setCalciumScale( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->calciumScale_ = value;
}

double SigNeur::getCalciumScale( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->calciumScale_;
}

void SigNeur::setDendInclude( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->dendInclude_ = value;
}

string SigNeur::getDendInclude( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dendInclude_;
}

void SigNeur::setDendExclude( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->dendExclude_ = value;
}

string SigNeur::getDendExclude( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dendExclude_;
}

void SigNeur::setChannelMap( const Conn* c, string val, const string& i )
{
	static_cast< SigNeur* >( c->data() )->channelMap_[ i ] = val;
}

string SigNeur::getChannelMap( Eref e, const string& i )
{
	SigNeur* sn = static_cast< SigNeur* >( e.data() );
	map< string, string >::iterator j = sn->channelMap_.find( i );
	if ( j != sn->channelMap_.end() )
		return j->second;
	return "";
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
	cout << "Building cell " << cellProto_.path() << endl;
	if ( !cellProto_.good() ) {
		cout << "SigNeur::build: " << c->target().name() << 
			" : Warning: Cell model prototype not defined.\n";
		return;
	}
	if ( !( spineProto_.good() || dendProto_.good() || somaProto_.good() ) ) {
		cout << "SigNeur::build: " << c->target().name() << 
			" : Warning: Unable to find any signaling models to use\n";
		return;
	}

	separateSpineSolvers_ = spineProto_.good() && dendProto_.good() && 
		spineMethod_ != dendMethod_;

	if ( !traverseCell( c->target() ) ) {
		cout << "SigNeur::build: " << c->target().name() << 
		cout << " : Warning: Unable to traverse cell\n";
		return;
	}

	schedule( c->target() );
}


bool SigNeur::traverseCell( Eref me )
{
	Element* cell = cellProto_.eref()->copy( me.e, "cell" );
	if ( !cell )
		return 0;
	cell_ = cell->id();

	vector< Id > compts;
	get< vector< Id > >( cell_.eref(), "childList", compts );
	if ( compts.size() == 0 )
		return 0;
	// Find soma. Use name and biggest dia.
	Id soma = findSoma( compts );

	// Build a tree of compts. Root is soma, this is first entry in 
	// tree_ vector. Depth-first.
	buildTree( soma, compts );

	// Figure out size of signaling model segments. Each elec compt must be
	// an integral number of signaling models.
	assignSignalingCompts();
	
	// Set up the signaling models
	makeSignalingModel( me );

	reportTree( volume_, xByL_ );

	makeCell2SigAdaptors( );

	makeSig2CellAdaptors( );

	return 1;
}

void SigNeur::schedule( Eref me )
{
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
	Shell::setClock( &c, 2, cellDt_, 2 );
	Shell::setClock( &c, 3, sigDt_, 0 );
	Shell::setClock( &c, 4, sigDt_, 1 );

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

	Shell::useClock( &c, "t3", "/sig/kinetics", "process" );
	Shell::useClock( &c, "t3", "/sig/kinetics/solve/hub", "process" );
	Shell::useClock( &c, "t3", "/sig/kinetics/solve/integ", "process" );

	Shell::useClock( &c, "t4", "/sig/cell/##[][TYPE==Adaptor]", "process" );
}
