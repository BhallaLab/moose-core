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
#include "../kinetics/Molecule.h"
#include "../kinetics/Reaction.h"
#include "../biophysics/Compartment.h"
#include "SigNeur.h"
#include "Adaptor.h"

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
	: 	
		numSpine_( 0 ), 
		numNeck_( 0 ), 
		numDend_( 0 ), 
		numSoma_( 0 ), 
		cellMethod_( "hsolve" ), 
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

/**
 * For now don't deal with taper
 */
unsigned int numSegments( Id compt, double lambda )
{
	double length = 0.0;
	// double dia = 0.0;
	assert( compt.good() );
	bool ret = get< double >( compt.eref(), "length", length );
	assert( ret );
	// ret = get< double >( compt.eref(), "diameter", dia );
	assert( ret );
	assert( length > 0.0 && lambda > 0.0 );
	return ( 1 + length / lambda );
}

// Count signaling compts, also subdivide for long dend compts.
void SigNeur::assignSignalingCompts()
{
	numSoma_ = 0;
	numDend_ = 0;
	numSpine_ = 0;
	numNeck_ = 0;
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		if ( i->category == SOMA ) {
			unsigned int ns = numSegments( i->compt, lambda_ );
			i->sigStart = numSoma_;
			i->sigEnd = numSoma_ = numSoma_ + ns;
		} else if ( i->category == DEND ) {
			unsigned int ns = numSegments( i->compt, lambda_ );
			i->sigStart = numDend_;
			i->sigEnd = numDend_ = numDend_ + ns;
			// cout << " " << numSegments;
		} else if ( i->category == SPINE ) {
			i->sigStart = numSpine_;
			i->sigEnd = ++numSpine_;
		} else if ( i->category == SPINE_NECK ) {
			++numNeck_;
		}
	}
	// cout << endl;
	// Now reposition the indices for the dends and spines, depending on
	// the numerical methods.
	// if ( dendMethod_ == "rk5" && somaMethod_ == dendMethod_ ) {
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == DEND ) {
				i->sigStart += numSoma_;
				i->sigEnd += numSoma_;
			}
		}
	// }
	// if ( dendMethod_ == "rk5" && spineMethod_ == dendMethod_ ) {
		unsigned int offset = numSoma_ + numDend_;
		for ( vector< TreeNode >::iterator i = tree_.begin(); 
				i != tree_.end(); ++i ) {
			if ( i->category == SPINE ) {
				i->sigStart += offset;
				i->sigEnd += offset;
			}
		}
	// }

	cout << "SigNeur: Tree size = " << tree_.size() << 
		", s=" << numSoma_ << 
		", d=" << numDend_ << 
		", sp=" << numSpine_ <<
		", neck=" << numNeck_ << endl;
}

void SigNeur::insertDiffusion( Element* base )
{
	static const double EPSILON = 1.0e-20;
	static const Finfo* childListFinfo = 
		initNeutralCinfo()->findFinfo( "childList" );
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* dFinfo = 
		initMoleculeCinfo()->findFinfo( "D" );
	static const Finfo* modeFinfo = 
		initMoleculeCinfo()->findFinfo( "mode" );
	static const Finfo* subFinfo = 
		initReactionCinfo()->findFinfo( "sub" );

	// Traverse all zero index children, find ones that have D > 0
	// Create an array of diffs on these children
	// Connect up to parent using One2OneMap
	// Connect up to next index parent using SimpleConn for now
	// Assign rates.
	if ( base == 0 )
		return;
	
	// Traverse children.
	vector< Id > kids;
	get< vector< Id > >( base, childListFinfo, kids );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
		insertDiffusion( i->eref().e );

	// Be sure to compete traversal before skipping this element.
	if ( !base->cinfo()->isA( initMoleculeCinfo() ) )
		return;
	
	// Ensure D > 0 
	double D = 0.0;
	bool ret = get< double >( base, dFinfo, D );
	if ( D <= EPSILON )
		return;
	int mode = 0;
	ret = get< int >( base, modeFinfo, mode );
	if ( mode != 0 ) // Should neither be buffered nor sumtotalled.
		return;

	// Create array of diffusion reactions.
	Id baseId = base->id();
	Id childId = Id::childId( baseId );
	Element* diff = Neutral::create( "Reaction", "diff", baseId, childId );

	assert( diff != 0 );
	// Connect up to parent
	ret = baseId.eref().add( reacFinfo->msg(), 
		childId.eref(), subFinfo->msg(), ConnTainer::Default );
	assert( ret );

	// assign rates.
	
}

/**
 * Return the diffusion reaction that is a child of this molecule, if
 * present. Otherwise return 0
 */
Element* findDiff( Element* pa )
{
	static const Finfo* lookupChildFinfo =
		initNeutralCinfo()->findFinfo( "lookupChild" );
	Id ret;
	string temp( "diff" );
	lookupGet< Id, string >( pa, lookupChildFinfo, ret, temp );

	if ( ret.good() )
		return ret();

	return 0;
}

/**
 * Utility function to do the diffusion calculations for the 
 * diffusing molecules m0 and m1, and the reaction diff.
 * Should be called at the point where the diffusion messages are set up.
 */
void diffCalc( double Dscale, Eref m0, Eref m1, Eref diff )
{
	static const Finfo* kfFinfo = 
		initReactionCinfo()->findFinfo( "kf" );
	static const Finfo* kbFinfo = 
		initReactionCinfo()->findFinfo( "kb" );
	static const Finfo* dFinfo = 
		initMoleculeCinfo()->findFinfo( "D" );
	static const Finfo* volFinfo = 
		initMoleculeCinfo()->findFinfo( "volumeScale" );

	double v0;
	double v1;
	double D0;
	double D1;
	bool ret = get< double >( m0, volFinfo, v0 );
	assert( ret && v0 > 0.0 );
	ret = get< double >( m0, dFinfo, D0 );
	assert( ret && D0 > 0.0 );
	ret = get< double >( m1, volFinfo, v1 );
	assert( ret && v1 > 0.0 );
	ret = get< double >( m1, dFinfo, D1 );
	assert( ret && D1 > 0.0 );
	double D = Dscale * ( D0 + D1 ) / 2.0;

	// volscale is conversion factor from # to uM: # / volscale = uM.
	// We need concs in #/m^3 to be consistent in unit terms.
	// #/( volscale) = 6e20 / m^3
	//
	// ( 1000 * # / 6e23 ) / vol(in m^3) = uM
	// # / ( 6e20 * vol ) = uM so volscale = 6e20 * vol
	// or vol = volscale / 6e20
	v0 /= 6.0e20;
	v1 /= 6.0e20;

	double kf = 0.0; // kf already set to Xarea / len
	ret = get< double >( diff, kfFinfo, kf );
	assert( ret && kf > 0.0 );
	kf *= D / v0;
	ret = set< double >( diff, kfFinfo, kf );
	assert( ret );

	double kb = 0.0; // kb already set to Xarea / len
	ret = get< double >( diff, kbFinfo, kb );
	assert( ret && kb > 0.0 );
	kb *= D / v1;
	ret = set< double >( diff, kbFinfo, kb );
	assert( ret );
}


void SigNeur::completeSomaDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = somaMap_.begin(); 
		i != somaMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numSoma_; ++j ) {
				if ( junctions[ j ] != UINT_MAX ) {
					assert( junctions[ j ] < i->second->numEntries() );
					Eref e2( diff, j );
					Eref e1( i->second, junctions[ j ] );
					bool ret = e1.add( reacFinfo->msg(), 
						e2, prdFinfo->msg(), 
						ConnTainer::Simple );
					assert( ret );
					Eref e0( i->second, j );
					diffCalc( Dscale_, e0, e1, e2 );
				}
			}
		}
	}
}

// void diffCalc( Eref m0, Eref m1, Eref diff )

void SigNeur::completeDendDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = dendMap_.begin(); 
		i != dendMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numDend_; ++j ) {
				unsigned int tgt = junctions[ j + numSoma_ ];
				if ( tgt < numSoma_ ) { // connect to soma, if mol present
					map< string, Element* >::iterator mol = 
						somaMap_.find( i->first );
					if ( mol != somaMap_.end() ) {
						assert( tgt < mol->second->numEntries() );
						Eref e2( diff, j );
						Eref e1( i->second, tgt );
						bool ret = e1.add( reacFinfo->msg(), 
							e2, prdFinfo->msg(), 
							ConnTainer::Simple );
						assert( ret );
						Eref e0( i->second, j );
						diffCalc( Dscale_, e0, e1, e2 );
					}
				} else if 
					( tgt >= numSoma_ && tgt < numDend_ + numSoma_ ) {
				// Look for other dend diffn compartments. Here tgt
				// is the same molecule, different index.
					tgt -= numSoma_;
					assert( tgt < i->second->numEntries() );
					Eref e2( diff, j );
					Eref e1( i->second, tgt );
					bool ret = e1.add( reacFinfo->msg(), 
						e2, prdFinfo->msg(), 
						ConnTainer::Simple );
					assert( ret );
					Eref e0( i->second, j );
					diffCalc( Dscale_, e0, e1, e2 );
				} else { // Should not connect into spine.
					assert( 0 );
				}
			}
		}
	}
}

void SigNeur::completeSpineDiffusion( 
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = spineMap_.begin(); 
		i != spineMap_.end(); ++i ) {
		Element* diff = findDiff( i->second );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numSpine_; ++j ) {
				unsigned int tgt = junctions[ j + numSoma_ + numDend_ ];
				if ( tgt >= numSoma_ && tgt < numSoma_ + numDend_ ) {
					tgt -= numSoma_; // Fix up offset into dend array.
					// connect to dend, if mol present
					map< string, Element* >::iterator mol = 
						dendMap_.find( i->first );
					if ( mol != dendMap_.end() ) {
						assert( tgt < mol->second->numEntries() );
						Eref e0( i->second, j ); // Parent spine molecule
						Eref e2( diff, j ); // Diffusion object
						Eref e1( mol->second, tgt ); //Target dend molecule
						bool ret = e1.add( reacFinfo->msg(), 
							e2, prdFinfo->msg(), 
							ConnTainer::Simple );
						assert( ret );
						diffCalc( Dscale_, e0, e1, e2 );
					}
				}
			}
		}
	}
}


/**
 * setAllVols traverses all signaling compartments  in the model and
 * assigns volumes.
 * This must be called before completeDiffusion because the vols
 * computed here are needed to compute diffusion rates.
 */
void SigNeur::setAllVols()
{
	for ( vector< TreeNode >::iterator i = tree_.begin();
		i != tree_.end(); ++i ) {
		for ( unsigned int j = i->sigStart; j < i->sigEnd; ++j ) {
			if ( i->category == SOMA ) {
				setComptVols( i->compt.eref(), 
					somaMap_, j, i->sigEnd - i->sigStart );
			} else if ( i->category == DEND ) {
				setComptVols( i->compt.eref(), 
					dendMap_, j - numSoma_, i->sigEnd - i->sigStart );
			} else if ( i->category == SPINE ) {
				setComptVols( i->compt.eref(), 
					spineMap_, j - ( numSoma_ + numDend_ ),
					i->sigEnd - i->sigStart );
			}
		}
	}
}

/**
 * This figures out dendritic segment dimensions. It assigns the 
 * volumeScale for each signaling compt, and puts Xarea / len into
 * each diffusion element for future use in setting up diffusion rates.
 */
void SigNeur::setComptVols( Eref compt, 
	map< string, Element* >& molMap,
	unsigned int index, unsigned int numSeg )
{
	static const Finfo* diaFinfo = 
		initCompartmentCinfo()->findFinfo( "diameter" );
	static const Finfo* lenFinfo = 
		initCompartmentCinfo()->findFinfo( "length" );
	static const Finfo* volFinfo = 
		initMoleculeCinfo()->findFinfo( "volumeScale" );
	static const Finfo* nInitFinfo = 
		initMoleculeCinfo()->findFinfo( "nInit" );
	static const Finfo* kfFinfo = 
		initReactionCinfo()->findFinfo( "kf" );
	static const Finfo* kbFinfo = 
		initReactionCinfo()->findFinfo( "kb" );
	assert( diaFinfo != 0 );
	assert( lenFinfo != 0 );
	assert( numSeg > 0 );
	double dia = 0.0;
	double len = 0.0;
	bool ret = get< double >( compt, diaFinfo, dia );
	assert( ret && dia > 0.0 );
	ret = get< double >( compt, lenFinfo, len );
	assert( ret && len > 0.0 );
	len /= numSeg;

	// Conversion factor from uM to #/m^3
	double volscale = len * dia * dia * ( PI / 4.0 ) * 6.0e20;
	double xByL = dia * dia * ( PI / 4.0 ) / len;
	
	// Set all the volumes. 
	for ( map< string, Element* >::iterator i = molMap.begin(); 
		i != molMap.end(); ++i ) {
		Element* de = findDiff( i->second );
		if ( !de )
			continue;
		Eref mol( i->second, index );
		Eref diff( de, index );
		double oldvol;
		double nInit;
		ret = get< double >( mol, volFinfo, oldvol );
		assert( ret != 0 && oldvol > 0.0 );
		ret = get< double >( mol, nInitFinfo, nInit );
		assert( ret );

		ret = set< double >( mol, volFinfo, volscale );
		assert( ret );
		ret = set< double >( mol, nInitFinfo, nInit * volscale / oldvol );
		assert( ret );
		ret = set< double >( diff, kfFinfo, xByL );
		assert( ret );
		ret = set< double >( diff, kbFinfo, xByL );
		assert( ret );
	}
}

/**
 * The first diffusion reaction (i.e., the one on sigStart) is the one
 * that crosses electrical compartment junctions. 
 *
 * For starters, we simply set the diameter at this and all other
 * diffusion reactions to that of the local electrical compartment.
 *
 * To represent a tapering dend cylinder, we could take the el dia as
 * that at sigStart, and the next compt dia as at sigEnd. But need
 * to rethink for branches.
 *
 * For spines, just use their spineNeck dimensions.
 *
 * For soma, ignore the soma dimensions except within it?
 *
 * It would be cleaner to take the el dia as the middle dia.
 *
void SigNeur::setDiffusionRates( 
	map< string, Element* >& somaMap, // Never needs to go off-map.
	map< string, Element* >& dendMap, // May go off-map to soma
	map< string, Element* >& spineMap, // Always goes off-map to dend.
	vector< unsigned int >& junctions )
{
	// flux = dN/dt = D.A.(C2 - C1) / L
	// We need to calculate: Vol of each compt
	// Then L = ( L1 + L2 ) / 2
	
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		for ( unsigned int j = i->sigStart; j < i->sigEnd; ++j ) {
			if ( category == SOMA ) {
				setRates( i->compt(), somaMap, j, numSoma_ );
			} else if ( category == DEND ) {
				setRates( i->compt(), dendMap, j - numSoma_, 
					i->sigEnd - i->sigStart );
			} else if ( category == SPINE ) {
				setSpineRates( i->compt(), i->parent, spineMap, 
					j - ( numSoma_ + numDend_ ) );
			}
		}
	}
	
}
 */

/**
 * Once the model is set up as an array, we need to go in and connect
 * diffusion reactions between compartments.
 * The 'base' is any element in the reaction tree. This routine will 
 * recursively traverse all descendants of 'base' to do its stuff.
 * 'junctions' holds the indices of connected compartments. 
 * The function looks for cases where the parent is a Molecule and the
 * base is a reaction named "diff"
 * Since the 'base' is typically an array element, it then goes through
 * all the array entries to set up the diffusion reactions.
void SigNeur::completeDiffusion( 
	map< string, Element* >& somaMap, // Never needs to go off-map.
	map< string, Element* >& dendMap, // May go off-map to soma
	map< string, Element* >& spineMap, // Always goes off-map to dend.
	vector< unsigned int >& junctions )
{
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	for ( map< string, Element* >::iterator i = somaMap.begin(); 
		i != somaMap.end(); ++i ) {
		Element* diff = findDiff( *i );
		if ( diff ) { // Connect up all diffn compts.
			for ( unsigned int j = 0; j < numSoma_; ++j ) {
				if ( junctions[ j ] != UINT_MAX ) {
					assert( junctions[ j ] < i->second->numEntries() );
					Eref e2( diff, j );
					Eref e1( i->second, junctions[ j ] );
					bool ret = e1.add( reacFinfo->msg(), 
						e2, prdFinfo->msg(), 
						ConnTainer::Simple );
					assert( ret );
				}
			}
		}
	}
}
*/

/*
void SigNeur::completeDiffusion( Element* parent, Element* base,
	unsigned int startIndex, vector< unsigned int >& junctions )
{
	static const Finfo* childSrcFinfo = 
		initNeutralCinfo()->findFinfo( "childSrc" );
	static const Finfo* reacFinfo = 
		initMoleculeCinfo()->findFinfo( "reac" );
	static const Finfo* prdFinfo = 
		initReactionCinfo()->findFinfo( "prd" );
	
	if ( parent == 0 || base == 0 )
		return;
	// Traverse children.
	const Msg* m = base->msg( childSrcFinfo->msg() );
	for ( vector< ConnTainer* >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		if ( (*i)->e2() != base )
			completeDiffusion( base, (*i)->e2(), startIndex, junctions );
		else
			completeDiffusion( base, (*i)->e1(), startIndex, junctions );
	}

	if ( parent->cinfo()->isA( initMoleculeCinfo() ) &&
		base->cinfo()->isA( initReactionCinfo() ) &&
		base->name() == "diff" ) {
		// Connect up to next index.
		assert ( base->numEntries()  + startIndex <= junctions.size() );
		for ( unsigned int i = 0; i < base->numEntries(); ++i ) {
			unsigned int j = i + startIndex;
			if ( junctions[j] < base->numEntries() ) {
				Eref e2( base, i );
				Eref e1( parent, junctions[ j ] - startIndex );
				bool ret = e1.add( reacFinfo->msg(), e2, prdFinfo->msg(), 
					ConnTainer::Simple );
				assert( ret );
			}
		}
	}
}
*/

/*
 * This function copies a signaling model. It first traverses the model and
 * inserts any required diffusion reactions into the model. These are
 * created as children of the molecule that diffuses, and are connected
 * up locally for one-half of the diffusion process. Subsequently the
 * system needs to connect up to the next compartment, to set up the 
 * other half of the diffusion. Also the last diffusion reaction
 * needs to have its rates nulled out.
 *
 * Returns the root element of the copy.
 * Kinid is destination of copy
 * proto is prototype
 * Name is the name to assign to the copy.
 * num is the number of duplicates needed.
 */
Element* SigNeur::copySig( Id kinId, Id proto, 
	const string& name, unsigned int num )
{
	Element* ret = 0;
	if ( proto.good() ) {
		Id lib( "/library" );
		/*
		Element* temp = Neutral::create( "Neutral", "temp", libId, 
			Id::childId( libId ) );
		*/
		ret = proto()->copy( lib(), name + ".msgs" );
		assert( ret );
		insertDiffusion( ret ); // Scan through putting in diffs.

		if ( num == 1 ) {
			ret = ret->copy( kinId(), name );
		} else if ( num > 1 ) {
			ret = ret->copyIntoArray( kinId, name, num );
		}
	}
	return ret;
}

void SigNeur::makeSignalingModel( Eref me )
{
	// Make kinetic manager on sigNeur
	// Make array copy of soma model.
	// Make array copy of dend model.
	// Make array copy of spine model.
	// Traverse tree, set up diffusion messages.
	// If any are nonzero, activate kinetic manager
	
	Element* kin = Neutral::create( "KineticManager", "kinetics", 
		me.id(), Id::childId( me.id() ) );
 	Id kinId = kin->id();
	// Each of these protos should be a simple Element neutral.
	Element* soma = copySig( kinId, somaProto_, "soma", numSoma_ );
	Element* dend = copySig( kinId, dendProto_, "dend", numDend_ );
	Element* spine = copySig( kinId, spineProto_, "spine", numSpine_ );

	if ( soma )
		soma_ = soma->id();
	if ( dend )
		dend_ = dend->id();
	if ( spine )
		spine_ = spine->id();

	// first soma indices, then dend, then spines.
	vector< unsigned int > junctions( 
		numSoma_ + numDend_ + numSpine_, UINT_MAX );
	buildDiffusionJunctions( junctions );
	buildMoleculeNameMap( soma, somaMap_ );
	buildMoleculeNameMap( dend, dendMap_ );
	buildMoleculeNameMap( spine, spineMap_ );

	/*
	for ( unsigned int j = 0; j < junctions.size(); ++j )
		cout << " " << j << "," << junctions[j];
	cout << endl;
	*/

	setAllVols();

	completeSomaDiffusion( junctions );
	completeDendDiffusion( junctions );
	completeSpineDiffusion( junctions );

	set< string >( kin, "method", dendMethod_ );
}

/**
 * Traverses the cell tree to work out where the diffusion reactions
 * must connect to each other. junction[i] is the index of the 
 * compartment connected to compartment[i]. The indexing of compartments
 * themselves is first the soma block, then the dend block, then the
 * spine block.
 */
void SigNeur::buildDiffusionJunctions( vector< unsigned int >& junctions )
{
	// TreeNode 0 is soma, has parent 0, rest should be a different compt. 
	// The first soma compartment should not connect to anything.
	// The next connects to the first soma, and so on.
	// The diffusion compartments start at the proximal compartment and
	// end at the distal compartment, with respect to the soma.
	// Because of the tree structure, sigStart must connect to the parent,
	// so that every branch point has a diffusion reaction for each branch.
	// Thus sigStart+1 must connect to sigStart, and so on.
	// We ignore diffusive coupling between sibling branches.
	//
	// Need to figure out how to put things at opposite ends of compt.
	for ( vector< TreeNode >::iterator i = tree_.begin(); 
		i != tree_.end(); ++i ) {
		// cout << i - tree_.begin() << "	" << i->compt.path() << ", p=" << i->parent << ", sig=" << i->sigStart << "," << i->sigEnd << endl;;

		
		// First we assign the links within the electrical compartment,
		//  this is easy.
		for ( unsigned int j = i->sigStart + 1; j < i->sigEnd; j++ )
			junctions[ j ] = j - 1;

		//////////////// Now we assign sigSTart ///////////////
		
		if ( i == tree_.begin() ) 
		// Skip sigStart for zero diffusion compartment
			continue;

		TreeNode* tn = &( tree_[ i->parent ] );
		// Attach spine not to neck, but to parent dend.
		if ( i->category == SPINE && 
			tn->category == SPINE_NECK && 
			i->sigStart != i->sigEnd ) {
			TreeNode* dend = &( tree_[ tn->parent ] );
			// Should position based on coordinates and length and numSeg
			// of parent dend. For now put it in middle
			assert( dend->sigStart != dend->sigEnd );
			junctions[ i->sigStart ] = ( dend->sigStart + dend->sigEnd )/ 2;
		} else if ( i->category == DEND && i->sigStart != i->sigEnd ) {
			// Attach it to last sig compt of parent electrical compartment
			// Actually should do something spatial here.
			junctions[ i->sigStart ] = tn->sigEnd - 1;
		}
	}
}

/**
 * Traverses the signaling tree to build a map of molecule Elements 
 * looked up by name.
 */
void SigNeur::buildMoleculeNameMap( Element* e,
	map< string, Element* >& molMap )
{
	static const Finfo* childSrcFinfo = 
		initNeutralCinfo()->findFinfo( "childSrc" );
	static const Finfo* parentFinfo = 
		initNeutralCinfo()->findFinfo( "parent" );
	if ( e == 0 )
		return;
	
	if ( e->cinfo()->isA( initMoleculeCinfo() ) ) {
		string ename = e->name();
		if ( ename == "kenz_cplx" ) {
			Id pa;
			get< Id >( e, parentFinfo, pa );
			Id grandpa;
			get< Id >( pa(), parentFinfo, grandpa );
			ename = grandpa()->name() + "__" + ename;
		}
		map< string, Element* >::iterator i = molMap.find( ename );
		if ( i != molMap.end() ) {
			cout << "buildMoleculeNameMap:: Warning: duplicate molecule: "
				<< i->second->id().path() << ", " << e->id().path() << endl;
		} else {
			molMap[ ename ] = e;
		}
	}
	// Traverse children.
	const Msg* m = e->msg( childSrcFinfo->msg() );
	for ( vector< ConnTainer* >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		if ( (*i)->e2() != e )
			buildMoleculeNameMap( (*i)->e2(), molMap );
		else
			buildMoleculeNameMap( (*i)->e1(), molMap );
	}
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

	makeCell2SigAdaptors( );

	makeSig2CellAdaptors( );

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

//////////////////////////////////////////////////////////////////////////

void adaptCa2Sig( TreeNode& t, 
	map< string, Element* >& m, Id caId, const string& mol )
{
	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		cout << "Adding adaptor from " << caId.path() << " to " <<
			i->second->name() << endl;
	}
}

void SigNeur::makeCell2SigAdaptors()
{
	static const Finfo* lookupChildFinfo = 
		initCompartmentCinfo()->findFinfo( "lookupChild" );
	for ( map< string, string >::iterator i = calciumMap_.begin();
		i != calciumMap_.end(); ++i ) {
		for ( vector< TreeNode >::iterator j = tree_.begin();
			j != tree_.end(); ++j ) {
			Id caId;
			bool ret = lookupGet< Id, string >( j->compt.eref(), lookupChildFinfo, caId, i->first );
			if ( ret && caId.good() ) {
				if ( j->category == SOMA ) {
					adaptCa2Sig( *j, somaMap_, caId, i->second );
				} else if ( j->category == DEND ) {
					adaptCa2Sig( *j, dendMap_, caId, i->second );
				} else if ( j->category == SPINE ) {
					adaptCa2Sig( *j, spineMap_, caId, i->second );
				}
			}
		}
	}
}

void adaptSig2Chan( TreeNode& t, 
	map< string, Element* >& m,
	unsigned int offset,
	const string& mol, Id chanId )
{
	static const Finfo* inputFinfo = 
		initAdaptorCinfo()->findFinfo( "input" );
	static const Finfo* outputFinfo = 
		initAdaptorCinfo()->findFinfo( "outputSrc" );
	
	// Not sure if these update when solved
	static const Finfo* molNumFinfo =  
		initMoleculeCinfo()->findFinfo( "nSrc" );

	// This isn't yet a separate destMsg. Again, issue with update.
	static const Finfo* gbarFinfo =  
		initMoleculeCinfo()->findFinfo( "Gbar" );

	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		Element* e = i->second;
		cout << "Adding adaptor from " << e->name() << " to " << 
			chanId.path() << endl;
		assert( t.sigStart >= offset );
		assert( t.sigEnd - offset <= e->numEntries() );

		// Create the adaptor
		Element* adaptor = Neutral::create( "Adaptor", "sig2chan",
			t.compt, Id::childId( t.compt ) );
		assert( adaptor != 0 );
		Eref adaptorE( adaptor, 0 );

		/*
		for ( unsigned int j = t.sigStart; j < t.sigEnd; ++j ) {
			// Connect up the adaptor.
			Eref molE( e, j - offset );
			bool ret = molE.add( molNumFinfo->msg(), adaptorE,
				inputFinfo->msg(), ConnTainer::Default );
			assert( ret );

			// Here we set the parameters of the adaptor.
		}
		ret = adaptorE.add( outputFinfo->msg(), chanId.eref(), 
			gbarFinfo->msg(), ConnTainer::Default );
		assert( ret );
		*/
	}
}


void SigNeur::makeSig2CellAdaptors()
{
	static const Finfo* lookupChildFinfo = 
		initCompartmentCinfo()->findFinfo( "lookupChild" );
	for ( map< string, string >::iterator i = channelMap_.begin();
		i != channelMap_.end(); ++i ) {
		for ( vector< TreeNode >::iterator j = tree_.begin();
			j != tree_.end(); ++j ) {
			Id chanId;
			bool ret = lookupGet< Id, string >( j->compt.eref(), lookupChildFinfo, chanId, i->second );
			if ( ret && chanId.good() ) {
				if ( j->category == SOMA ) {
					adaptSig2Chan( *j, somaMap_, 0, i->first, chanId );
				} else if ( j->category == DEND ) {
					adaptSig2Chan( *j, dendMap_, numSoma_, 
						i->first, chanId );
				} else if ( j->category == SPINE ) {
					adaptSig2Chan( *j, spineMap_, numDend_ + numSoma_,
						i->first, chanId );
				}
			}
		}
	}
}
