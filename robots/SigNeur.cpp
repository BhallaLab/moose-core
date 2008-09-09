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
#include "setgetLookup.h"
#include "../element/Neutral.h"
#include "SigNeur.h"


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
	
	static Cinfo sigNeurCinfo(
		"SigNeur",
		"Upinder S. Bhalla, 2007, NCBS",
		"SigNeur: Multiscale simulation setup object for doing combined electrophysiological and signaling models of neurons. Takes the geometry from the neuronal model and sets up diffusion between signaling models to fit in this geometry. Arranges interfaces between channel conductances and molecular species representing channels. Also interfaces calcium conc in the two kinds of model.",
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
	: 	cellMethod_( "hsolve" ), 
		spineMethod_( "rk5" ), 
		dendMethod_( "rk5" ), 
		somaMethod_( "rk5" ), 
		Dscale_( 1.0 ),
		lambda_( 10.0e-6 ),
		parallelMode_( 0 ),
		updateStep_( 1.0 ),
		calciumScale_( 1.0 )
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
	if ( !traverseCell( c->target() ) ) {
		cout << "SigNeur::build: " << c->target().name() << 
		cout << " : Warning: Unable to traverse cell\n";
		return;
	}
}

// Count signaling compts, also subdivide for long dend compts.
void SigNeur::assignSignalingCompts()
{
	unsigned int numSoma = 0;
	unsigned int numDend = 0;
	unsigned int numSpine = 0;
	unsigned int numNeck = 0;
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		if ( i->category == SOMA ) {
			i->sigStart = numSoma;
			i->sigEnd = ++numSoma;
		} else if ( i->category == DEND ) {
			double length;
			get< double >( i->compt.eref(), "length", length );
			unsigned int numSegments = 1 + length / lambda_;
			i->sigStart = numDend;
			i->sigEnd = numDend = numDend + numSegments;
			// cout << " " << numSegments;
		} else if ( i->category == SPINE ) {
			i->sigStart = numSpine;
			i->sigEnd = ++numSpine;
		} else if ( i->category == SPINE_NECK ) {
			++numNeck;
		}
	}
	// cout << endl;
	// Now reposition the indices for the dends and spines, depending on
	// the numerical methods.
	if ( dendMethod_ == "rk5" && somaMethod_ == dendMethod_ ) {
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == DEND ) {
				i->sigStart += numSoma;
				i->sigEnd += numSoma;
			}
		}
	}
	if ( dendMethod_ == "rk5" && spineMethod_ == dendMethod_ ) {
		unsigned int offset = numSoma + numDend;
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == SPINE ) {
				i->sigStart += offset;
				i->sigEnd += offset;
			}
		}
	}

	cout << "SigNeur: Tree size = " << tree_.size() << ", s=" << numSoma << 
		", d=" << numDend << ", sp=" << numSpine <<
		", neck=" << numNeck << endl;
}

void SigNeur::makeSignalingModel()
{
	;
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
	makeSignalingModel();

	return 1;
}

Id SigNeur::findSoma( const vector< Id >& compts )
{
	double maxDia = 0;
	Id maxCompt;
	vector< Id > somaCompts; // Theoretically possible to have an array.
	for ( vector< Id >::const_iterator i = compts.begin(); 
		i != compts.end(); ++i )
	{
		string className = i->eref()->className();
		if ( className == "Compartment" || className == "SymCompartment" ) {
			string name = i->eref().e->name();
			if ( name == "soma" || name == "Soma" || name == "SOMA" )
				somaCompts.push_back( *i );
			double dia;
			get< double >( i->eref(), "diameter", dia );
			if ( dia > maxDia )
				maxCompt = *i;
		}
	}
	if ( somaCompts.size() == 1 ) // First, go by name.
		return somaCompts[0];
	if ( somaCompts.size() == 0 & maxCompt.good() ) //if no name, use maxdia
		return maxCompt;
	if ( somaCompts.size() > 1 ) { // Messy but unlikely cases.
		if ( maxCompt.good() ) {
			if ( find( somaCompts.begin(), somaCompts.end(), maxCompt ) != somaCompts.end() )
				return maxCompt;
			else
				cout << "Error, soma '" << somaCompts.front().path() << 
					"' != biggest compartment '" << maxCompt.path() << 
					"'\n";
		}
		return somaCompts[0]; // Should never happen, but an OK response.
	}
	cout << "Error: SigNeur::findSoma failed to find soma\n";
	return Id();
}

void SigNeur::buildTree( Id soma, const vector< Id >& compts )
{
	const Cinfo* symCinfo = Cinfo::find( "SymCompartment" );
	assert( symCinfo != 0 );
	const Finfo* axialFinfo;
	const Finfo* raxialFinfo;
	if ( soma.eref().e->cinfo() == symCinfo ) {
		axialFinfo = symCinfo->findFinfo( "raxial1" );
		raxialFinfo = symCinfo->findFinfo( "raxial2" );
	} else {
		const Cinfo* asymCinfo = Cinfo::find( "Compartment" );
		assert( asymCinfo != 0 );
		axialFinfo = asymCinfo->findFinfo( "axial" );
		raxialFinfo = asymCinfo->findFinfo( "raxial" );
	}
	assert( axialFinfo != 0 );
	assert( raxialFinfo != 0 );
	
	// Soma may be in middle of messaging structure for cell, so we need
	// to traverse both ways. But nothing below soma should 
	// change direction in the traversal.
	innerBuildTree( 0, soma.eref(), soma.eref(), 
		axialFinfo->msg(), raxialFinfo->msg() );
	// innerBuildTree( 0, soma.eref(), soma.eref(), raxialFinfo->msg() );
}

void SigNeur::innerBuildTree( unsigned int parent, Eref paE, Eref e, 
	int msg1, int msg2 )
{
	unsigned int paIndex = tree_.size();
	TreeNode t( e.id(), parent, guessCompartmentCategory( e ) );
	tree_.push_back( t );
	// cout << e.name() << endl;
	Conn* c = e->targets( msg1, e.i );

	// Things are messy here because src/dest directions are flawed
	// in Element::targets.
	// The parallel moose fixes this mess, simply by checking against
	// which the originating element is. Here we need to do the same
	// explicitly.
	for ( ; c->good(); c->increment() ) {
		Eref tgtE = c->target();
		if ( tgtE == e )
			tgtE = c->source();
		if ( !( tgtE == paE ) ) {
			// cout << "paE=" << paE.name() << ", e=" << e.name() << ", msg1,2= " << msg1 << "," << msg2 << ", src=" << c->source().name() << ", tgt= " << tgtE.name() << endl;
			innerBuildTree( paIndex, e, tgtE, msg1, msg2 );
		}
	}
	delete c;
	c = e->targets( msg2, e.i );
	for ( ; c->good(); c->increment() ) {
		Eref tgtE = c->target();
		if ( tgtE == e )
			tgtE = c->source();
		if ( !( tgtE == paE ) ) {
			// cout << "paE=" << paE.name() << ", e=" << e.name() << ", msg1,2= " << msg1 << "," << msg2 << ", src=" << c->source().name() << ", tgt= " << tgtE.name() << endl;
			innerBuildTree( paIndex, e, tgtE, msg1, msg2 );
		}
	}
	delete c;
}


CompartmentCategory SigNeur::guessCompartmentCategory( Eref e )
{
	if ( e.e->name().find( "spine" ) != string::npos ||
		e.e->name().find( "Spine" ) != string::npos ||
		e.e->name().find( "SPINE" ) != string::npos )
	{
		if ( e.e->name().find( "neck" ) != string::npos ||
			e.e->name().find( "Neck" ) != string::npos ||
			e.e->name().find( "NECK" ) != string::npos ||
			e.e->name().find( "shaft" ) != string::npos ||
			e.e->name().find( "Shaft" ) != string::npos ||
			e.e->name().find( "SHAFT" ) != string::npos
		)
			return SPINE_NECK;
		else
			return SPINE;
	}
	if ( e.e->name().find( "soma" ) != string::npos ||
		e.e->name().find( "Soma" ) != string::npos ||
		e.e->name().find( "SOMA" ) != string::npos)
	{
		return SOMA;
	}
	return DEND;
}
