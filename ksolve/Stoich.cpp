/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <map>
#include <algorithm>

#include "moose.h"
#include "../element/Wildcard.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "InterSolverFlux.h"
#include "Stoich.h"
#include "kinetics/Molecule.h"
#include "kinetics/Reaction.h"
#include "kinetics/Enzyme.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

const double RateTerm::EPSILON = 1.0e-6;

	// Limits the k1, but unfortunately there is a volume scaling here.
	// for reference, if vol = 1e-15, k1 ~ 1e-6 for a Km of 1.
const double Stoich::EPSILON = 1.0e-12; 

const Cinfo* initStoichCinfo()
{
	// connects to the KineticHub object
	static Finfo* hubShared[] =
	{
		new SrcFinfo( "rateTermInfoSrc", 
			Ftype3< vector< RateTerm* >*, KinSparseMatrix*, bool >::global()
		),
		new SrcFinfo( "rateSizeSrc", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global()
		),
		new SrcFinfo( "molSizeSrc", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global()
		),
		new SrcFinfo( "molConnectionSrc",
			Ftype3< vector< double >* , 
				vector< double >* , 
				vector< Eref >*  
				>::global(),
				"This one is a bit ugly. It sends the entire S_ and Sinit_  arrays over"
		),
		new SrcFinfo( "reacConnectionSrc",
			Ftype2< unsigned int, Eref >::global()
		),
		new SrcFinfo( "enzConnectionSrc",
			Ftype2< unsigned int, Eref >::global()
		),
		new SrcFinfo( "mmEnzConnectionSrc",
			Ftype2< unsigned int, Eref >::global()
		),
		new SrcFinfo( "completeSetupSrc",
			Ftype1< string >::global()
		),
		new SrcFinfo( "clearSrc",
			Ftype0::global()
		),
		new DestFinfo( "setMolN", 
			Ftype2< double, unsigned int >::global(),
			RFCAST( &Stoich::setMolN )
		),
		new DestFinfo( "setBuffer", 
			Ftype2< int, unsigned int >::global(),
			RFCAST( &Stoich::setBuffer ),
			"Assigns dynamic buffers. First arg is mode and second is the molecule index."
		),
	};
	static Finfo* integrateShared[] =
	{
		new DestFinfo( "reinit", Ftype0::global(),
			&Stoich::reinitFunc ),
		new DestFinfo( "integrate",
			Ftype2< vector< double >* , double >::global(),
			RFCAST( &Stoich::integrateFunc ) ),
		new SrcFinfo( "allocate",
			Ftype1< vector< double >* >::global() ),
	};
	static Finfo* gslShared[] =
	{
		new DestFinfo( "reinit", Ftype0::global(),
			&Stoich::reinitFunc ),
		new SrcFinfo( "assignStoich",
			Ftype1< void* >::global() ),
		new SrcFinfo( "setMolNsrc",
			Ftype2< double, unsigned int >::global() ),
		new DestFinfo( "requestY", Ftype0::global(),
			&Stoich::requestY ),
		new SrcFinfo( "assignYsrc",
			Ftype1< double* >::global() ),
	};

	/**
	 * These are the fields of the stoich class
	 */
	static Finfo* stoichFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "nMols", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNmols ), 
			&dummyFunc
		),
		new ValueFinfo( "nVarMols", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNvarMols ), 
			&dummyFunc
		),
		new ValueFinfo( "nSumTot", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNsumTot ), 
			&dummyFunc
		),
		new ValueFinfo( "nBuffered", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNbuffered ), 
			&dummyFunc
		),
		new ValueFinfo( "nReacs", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNreacs ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNenz ), 
			&dummyFunc
		),
		new ValueFinfo( "nMMenz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNmmEnz ), 
			&dummyFunc
		),
		new ValueFinfo( "nExternalRates", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getNexternalRates ), 
			&dummyFunc
		),
		new ValueFinfo( "useOneWayReacs", 
			ValueFtype1< bool >::global(),
			GFCAST( &Stoich::getUseOneWayReacs ), 
			RFCAST( &Stoich::setUseOneWayReacs )
		),
		new ValueFinfo( "path", 
			ValueFtype1< string >::global(),
			GFCAST( &Stoich::getPath ), 
			RFCAST( &Stoich::setPath )
		),
		new ValueFinfo( "pathVec", 
			ValueFtype1< vector< Id > >::global(),
			GFCAST( &Stoich::getPathVec ), 
			RFCAST( &Stoich::setPathVec )
		),
		new ValueFinfo( "rateVectorSize", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getRateVectorSize ), 
			&dummyFunc
		),
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		/*
		new DestFinfo( "rebuild", 
			Ftype0::global(),
			RFCAST( &Stoich::rebuild )
		),
		*/
		new DestFinfo( "rescaleVolume", 
			Ftype1< double >::global(),
			RFCAST( &Stoich::rescaleVolume ),
			"Scales the volume of the model by the specified ratio."
		),

		new DestFinfo( "makeFlux", 
			Ftype3< string, 
				vector< unsigned int >, vector< double > >::global(),
			RFCAST( &Stoich::makeFlux ),
			"Sets up an InterSolverFlux stub."
			"makeFlux( stubName, molIndices, fluxRates )"
			"The Stoich adds another entry to its flux_ vector, and"
			"this becomes a new child object that acts as the stub."
		),
		new DestFinfo( "startFromCurrentConcs", 
			Ftype0::global(),
			RFCAST( &Stoich::startFromCurrentConcs ),
			"Copies over the current state of the kinetic system to use"
			"for initial conditions. Sinit = S"
		),
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "hub", hubShared, 
				sizeof( hubShared )/ sizeof( Finfo* ),
					"Several messages that connect to the KineticHub" ),
		new SharedFinfo( "integrate", integrateShared, 
				sizeof( integrateShared )/ sizeof( Finfo* ),
					"Messages that connect to the KineticIntegrator" ),
		new SharedFinfo( "gsl", gslShared, 
				sizeof( gslShared )/ sizeof( Finfo* ),
					"Messages that connect to the GslIntegrator object" ),

	};
	
	static string doc[] =
	{
		"Name", "Stoich",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Stoich: Sets up stoichiometry matrix based calculations from a wildcard path for "
				"the reaction system. Knows how to compute derivatives for most common things, also "
				"knows how to handle special cases where the object will have to do its own "
				"computation. Generates a stoichiometry matrix,which is useful for lots of other "
				"operations as well."
				"Also provides child stub objects to act as a hook"
				"for inter-solver flow of molecules"
	};
	
	static Cinfo stoichCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos )/sizeof(Finfo *),
		ValueFtype1< Stoich >::global()
	);

	return &stoichCinfo;
}

static const Cinfo* stoichCinfo = initStoichCinfo();

static const Slot rateTermInfoSlot =
	initStoichCinfo()->getSlot( "hub.rateTermInfoSrc" );
static const Slot rateSizeSlot =
	initStoichCinfo()->getSlot( "hub.rateSizeSrc" );
static const Slot molSizeSlot =
	initStoichCinfo()->getSlot( "hub.molSizeSrc" );
static const Slot molConnectionSlot =
	initStoichCinfo()->getSlot( "hub.molConnectionSrc" );
static const Slot reacConnectionSlot =
	initStoichCinfo()->getSlot( "hub.reacConnectionSrc" );
static const Slot enzConnectionSlot =
	initStoichCinfo()->getSlot( "hub.enzConnectionSrc" );
static const Slot mmEnzConnectionSlot =
	initStoichCinfo()->getSlot( "hub.mmEnzConnectionSrc" );
static const Slot completeSetupSlot =
	initStoichCinfo()->getSlot( "hub.completeSetupSrc" );
static const Slot clearSlot =
	initStoichCinfo()->getSlot( "hub.clearSrc" );
static const Slot allocateSlot =
	initStoichCinfo()->getSlot( "integrate.allocate" );
static const Slot assignStoichSlot =
	initStoichCinfo()->getSlot( "gsl.assignStoich" );
static const Slot setMolNslot =
	initStoichCinfo()->getSlot( "gsl.setMolNsrc" );
static const Slot assignYslot =
	initStoichCinfo()->getSlot( "gsl.assignYsrc" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Stoich::Stoich()
{
	nMols_ = 0;
	nVarMols_ = 0;
	nVarMolsBytes_ = 0;
	nSumTot_ = 0;
	nBuffered_ = 0;
	nReacs_ = 0;
	nEnz_ = 0;
	nMmEnz_ = 0;
	nExternalRates_ = 0;
	useOneWayReacs_ = 0;
	lasty_ = 0;
}

Stoich::~Stoich()
{
	for ( vector< RateTerm* >::iterator i = rates_.begin(); 
		i != rates_.end(); ++i )
		delete (*i);
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int Stoich::getNmols( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nMols_;
}

unsigned int Stoich::getNvarMols( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nVarMols_;
}

unsigned int Stoich::getNsumTot( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nSumTot_;
}

unsigned int Stoich::getNbuffered( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nBuffered_;
}

unsigned int Stoich::getNreacs( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nReacs_;
}

unsigned int Stoich::getNenz( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nEnz_;
}

unsigned int Stoich::getNmmEnz( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nMmEnz_;
}

unsigned int Stoich::getNexternalRates( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->nExternalRates_;
}
void Stoich::setUseOneWayReacs( const Conn* c, int value ) {
	static_cast< Stoich* >( c->data() )->useOneWayReacs_ = value;
}

bool Stoich::getUseOneWayReacs( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->
		useOneWayReacs_;
}

string Stoich::getPath( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->path_;
}
void Stoich::setPath( const Conn* c, string value ) {
	static_cast< Stoich* >( c->data() )->localSetPath( c->target(), value);
}

vector< Id > Stoich::getPathVec( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->pathVec_;
}
void Stoich::setPathVec( const Conn* c, vector< Id > value ) {
	static_cast< Stoich* >( c->data() )->localSetPathVec( 
		c->target(), value);
}

unsigned int Stoich::getRateVectorSize( Eref e ) {
	return static_cast< const Stoich* >( e.data() )->rates_.size();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// Static func
void Stoich::reinitFunc( const Conn* c )
{
	Stoich* s = static_cast< Stoich* >( c->data() );
	s->S_ = s->Sinit_;
	/*
	cout << "Stoich::reinitFunc: Assigning S_. \nS_ = ( ";
	for ( unsigned int i = 0; i < s->S_.size(); ++i )
		cout << s->S_[i] << ", ";
	cout << " ), Sinit_ = ( ";
	for ( unsigned int i = 0; i < s->Sinit_.size(); ++i )
		cout << s->Sinit_[i] << ", ";
	cout << " )\n";
	*/


	// send1< vector< double >* >( e, allocateSlot, &s->S_ );
	// send1< void* >( e, assignStoichSlot, e->data() );
	s->lasty_ = 0;
	s->nCopy_ = 0;
	s->nCall_ = 0;
}

// static func
void Stoich::integrateFunc( const Conn* c, vector< double >* v, double dt )
{
	Stoich* s = static_cast< Stoich* >( c->data() );
	s->updateRates( v, dt );
}


/**
 * Relays a 'y' assignment request to the GSL integrator, or whatever
 * else is appropriate.
 */
void Stoich::setMolN( const Conn* c, double y, unsigned int i )
{
	static_cast< Stoich* >( c->data() )->innerSetMolN( c, y, i );
}

/**
 * virtual function, for the GssaStoich we have to do much more.
 */
void Stoich::innerSetMolN( const Conn* c, double y, unsigned int i )
{
	if ( i < nVarMols_ )
		send2< double, unsigned int>( c->target(), setMolNslot, y, i );
}

void Stoich::rescaleVolume( const Conn* c, double ratio ) {
	static_cast< Stoich* >( c->data() )->innerRescaleVolume( ratio );
}

void Stoich::innerRescaleVolume( double ratio )
{
	for ( vector< double >::iterator i = S_.begin(); i != S_.end(); ++i)
		*i *= ratio;
	for ( vector< double >::iterator i = Sinit_.begin(); i != Sinit_.end(); ++i)
		*i *= ratio;
	for ( vector< RateTerm* >::iterator i = rates_.begin(); i != rates_.end(); ++i)
		( *i )->rescaleVolume( ratio );
}

void Stoich::setBuffer( const Conn* c, int mode, unsigned int mol )
{
	static_cast< Stoich* >( c->data() )->innerSetBuffer( mode, mol );
}

void Stoich::innerSetBuffer( int mode, unsigned int mol )
{
	if ( mol >= ( nVarMols_ + nSumTot_ ) ) {
		if ( mode != 4 ) {
			cout << "Stoich::innerSetBuffer: Warning: Cannot change buffer state of predefined buffer molecule\n";
		}
		// In either case, just return at this point.
		return;
	}
	vector< unsigned int >::iterator pos = 
		find( dynamicBuffers_.begin(), dynamicBuffers_.end(), mol );
	if ( mode == 4 ) { // Buffering on. 
		if ( pos == dynamicBuffers_.end() )
			dynamicBuffers_.push_back( mol );
	} else { // buffering off
		if ( pos != dynamicBuffers_.end() )
			dynamicBuffers_.erase( pos );
	}
}

void Stoich::makeFlux( const Conn* c, 
	string stubName, vector< unsigned int >molIndices, 
	vector< double > fluxRates )
{
	static_cast< Stoich* >( c->data() )->innerMakeFlux(
		c->target(), stubName, molIndices, fluxRates );
}

/**
 * Puts the data into a new entry in the flux vector, and creates
 * a stub child for handling the messages to and from the entry.
 */
void Stoich::innerMakeFlux( Eref e,
	string stubName, vector< unsigned int >molIndices, 
	vector< double > fluxRates )
{
	assert( molIndices.size() == fluxRates.size() );
	vector< double* > fluxMolPtrs;
	for ( vector< unsigned int >::iterator i = molIndices.begin();
		i != molIndices.end(); ++i ) {
		assert( *i < S_.size() );
		fluxMolPtrs.push_back( &S_[ *i ] );
	}
	InterSolverFlux* data = 
		new InterSolverFlux( fluxMolPtrs, fluxRates ); 
	flux_.push_back( data );
	const Cinfo * ic = initInterSolverFluxCinfo();
	Element* f = ic->create( Id::scratchId(), stubName,
		static_cast< void* >( data ), 1 );
	e.add( "childSrc", f, "child" );
}

void Stoich::startFromCurrentConcs( const Conn* c ) {
	static_cast< Stoich* >( c->data() )->innerStartFromCurrentConcs();
}

void Stoich::innerStartFromCurrentConcs()
{
	assert( Sinit_.size() == S_.size() );
	Sinit_.assign( S_.begin(), S_.end() );

	for ( map< Eref, unsigned int >::iterator i = molMap_.begin();
		i != molMap_.end(); ++i ) {
		assert( i->second < S_.size() );
		Eref e = i->first;
		Molecule* m = static_cast< Molecule* >( e.data() );
		m->localSetNinit( S_[ i->second ] );
		/*
		SetConn c( i->first );
		Molecule::setConc( &c, S_[ i->second ] );
		*/
	}
}

/// Send S to gsl to update.
void Stoich::requestY( const Conn* c )
{
#ifdef USE_GSL
	double* s = static_cast< Stoich* >( c->data() )->S();
	send1< double* >( c->target(), assignYslot, s );
#endif // USE_GSL
}
///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

unsigned int countRates( Eref e, bool useOneWayReacs )
{
	if ( e.e->cinfo()->isA( initReactionCinfo() ) ) {
		if ( useOneWayReacs)
			return 2;
		else
			return 1;
	}
	if ( e.e->cinfo()->isA( initEnzymeCinfo() ) ) {
		bool enzmode = 0;
		bool isOK = get< bool >( e, "mode", enzmode );
		assert( isOK );
		if ( enzmode == 0 ) {
			if ( useOneWayReacs )
				return 3;
			else
				return 2;
		} else { 
			return 1;
		}
	}
	return 0;
}

void Stoich::clear( Eref stoich )
{
	// cout << "Sending clear signal for " << stoich->name() << "\n" << flush;
	send0( stoich, clearSlot ); // Sends command to KineticHub to clear
			// out the old messages to solved objects, and unzombify them
	// cout << "Clear signal sent\n" << flush;

	nMols_ = 0;
	nVarMols_ = 0;
	nSumTot_ = 0;
	nBuffered_ =0;
	nReacs_ = 0;
	nEnz_ = 0;
	nMmEnz_ = 0;
	nExternalRates_ = 0;
	S_.resize( 0 );
	Sinit_.resize( 0 );
	v_.resize( 0 );
	rates_.resize( 0, 0 );
	sumTotals_.resize( 0 );
	pathVec_.resize( 0 );
	path2mol_.resize( 0 );
	mol2path_.resize( 0 );
	molMap_.clear( );
#ifdef DO_UNIT_TESTS
	reacMap_.clear( );
#endif // DO_UNIT_TESTS
	nVarMolsBytes_ = 0;
	nCopy_ = 0;
	nCall_ = 0;
}

void Stoich::localSetPath( Eref stoich, const string& value )
{
	vector< Id > ret;
	wildcardFind( value, ret );
	this->localSetPathVec( stoich, ret ); //Pass down to derived classes
	path_ = value;
}

void Stoich::localSetPathVec( Eref stoich, vector< Id >& value)
{
	path_ = "assigned from pathVec";
	clear( stoich );
	pathVec_ = value;
	if ( pathVec_.size() > 0 ) {
		// Pass to derived classes
		this->rebuildMatrix( stoich, pathVec_ );

		// This first target is for any Kintegrator objects
		send1< vector< double >* >( stoich, allocateSlot, &S_ );

		// The second target is for GSL integrator types.
		send1< void* >( stoich, assignStoichSlot, stoich.data() );
	} else {
		cout << "No objects to simulate in pathVec sized " << 
			pathVec_.size() << "\n";
	}
}

/**
 * Virtual function to make the data structures from the 
 * object oriented specification of the signaling network.
 */
void Stoich::rebuildMatrix( Eref stoich, vector< Id >& ret )
{
	static const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	vector< Id >::iterator i;
	vector< Eref > varMolVec;
	vector< Eref > bufVec;
	vector< Eref > sumTotVec;
	int mode;
	bool isOK;
	unsigned int numRates = 0;
	sort( ret.begin(), ret.end() );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )()->cinfo()->isA( molCinfo ) ) {
			isOK = get< int >( ( *i )(), "mode", mode );
			assert( isOK );
			if ( mode == 0 ) {
				varMolVec.push_back( i->eref() );
			} else if ( mode == 4 ) {
				bufVec.push_back( i->eref() );
			} else {
				sumTotVec.push_back( i->eref() );
			}
		} else {
			numRates += countRates( i->eref(), useOneWayReacs_ );
		}
	}
	/*
	 * We set up the stoichiometry matrix with _all_ the molecules
	 * here. Later we only update those rows (molecules) that are 
	 * variable, but for other kinds of matrix-based analyses
	 * it is a good thing to have the
	 * whole lot accessible.
	 */
	setupMols( stoich, varMolVec, bufVec, sumTotVec );
	N_.setSize( nMols_, numRates );
	v_.resize( numRates, 0.0 );
	send3< vector< RateTerm* >*, KinSparseMatrix*, bool >( 
		stoich, rateTermInfoSlot, &rates_, &N_, useOneWayReacs_ );
	int nReac = 0;
	int nEnz = 0;
	int nMmEnz = 0;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( i->eref().e->cinfo()->isA( initReactionCinfo() ) ) {
			nReac++;
		} else if ( i->eref().e->cinfo()->isA( initEnzymeCinfo() ) ) {
			bool enzmode = 0;
			isOK = get< bool >( i->eref(), "mode", enzmode );
			assert( isOK );
			if ( enzmode == 0 )
				nEnz++;
			else
				nMmEnz++;
		}
	}
	// cout << "RebuildMatrix: Intermedate setup: " << nVarMols_ << "," << nReac << ", " << nEnz << ", " << nMmEnz << endl;
	send3< unsigned int, unsigned int, unsigned int >(
			stoich, rateSizeSlot, nReac, nEnz, nMmEnz );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		const string& cn = i->eref().e->className();
		if ( cn == "Reaction" ) {
			addReac( stoich, i->eref() );
		} else if ( cn == "Enzyme" ) {
			bool enzmode = 0;
			isOK = get< bool >( i->eref(), "mode", enzmode );
			assert( isOK );
			if ( enzmode == 0 )
				addEnz( stoich, i->eref() );
			else
				addMmEnz( stoich, i->eref() );
		} else if ( cn == "Table" ) {
			addTab( stoich, ( *i )() );
		} else if ( cn == "Neutral" ) {
			// cout << (*i)()->name() << " is a Neutral\n";
		} else if ( cn == "GslIntegrator" ||
			cn == "Kintegrator" ||
			cn == "KineticHub" ||
			cn == "KinCompt" ||
			cn == "Stoich" )
		{
			// Ignore these
			;
		} else if ( !( *i )()->cinfo()->isA( molCinfo ) ) {
			addRate( stoich, i->eref() );
		}
	}
	setupReacSystem( stoich );
	// cout << "RebuildMatrix: later setup: " << nVarMols_ << "," << nReac << ", " << nEnz << ", " << nMmEnz << endl;
}


void Stoich::setupMols(
	Eref e,
	vector< Eref >& varMolVec,
	vector< Eref >& bufVec,
	vector< Eref >& sumTotVec
	)
{
	const Finfo* nInitFinfo = Cinfo::find( "Molecule" )->
		findFinfo( "nInit" );
	// Field nInitField = Cinfo::find( "Molecule" )->field( "nInit" );
	vector< Eref >::iterator i;
	vector< Eref >elist;
	unsigned int j = 0;
	double nInit;
	nVarMols_ = varMolVec.size();
	nVarMolsBytes_ = nVarMols_ * sizeof( double );
	nSumTot_  = sumTotVec.size();
	nBuffered_ = bufVec.size();
	nMols_ = nVarMols_ + nSumTot_ + nBuffered_;
	S_.resize( nMols_ );
	Sinit_.resize( nMols_ );
	for ( i = varMolVec.begin(); i != varMolVec.end(); i++ ) {
		bool ret = get< double >( *i, nInitFinfo, nInit );
		assert( ret );
		assert( !isnan( nInit ) );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = sumTotVec.begin(); i != sumTotVec.end(); i++ ) {
		bool ret = get< double >( *i, nInitFinfo, nInit );
		assert( ret );
		assert( !isnan( nInit ) );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = bufVec.begin(); i != bufVec.end(); i++ ) {
		bool ret = get< double >( *i, nInitFinfo, nInit );
		assert( ret );
		assert( !isnan( nInit ) );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = sumTotVec.begin(); i != sumTotVec.end(); i++ ) {
		addSumTot( *i );
	}
	send3< unsigned int, unsigned int, unsigned int >(
		e, molSizeSlot, nVarMols_, nBuffered_, nSumTot_ );
	// molSizesSrc_.send( nVarMols_, nBuffered_, nSumTot_ );
	// molConnectionsSrc_.send( &S_, &Sinit_, &elist );
	send3< vector< double >* , vector< double >* , vector< Eref >*  >(
		e, molConnectionSlot, &S_, &Sinit_, &elist );
}

void Stoich::addSumTot( Eref e )
{
	vector< const double* > mol;
	// vector< Eref > tab;
	if ( findTargets( e, "sumTotal", mol ) ) {
		map< Eref, unsigned int >::iterator j = molMap_.find( e );
		assert( j != molMap_.end() );
		SumTotal st( &S_[ j->second ], mol );
		sumTotals_.push_back( st );
	}
}

// This replaces findIncoming and findReactants.
bool Stoich::findTargets(
	Eref e, const string& msgFieldName, vector< const double* >& ret )
{
	Conn* c = e.e->targets( msgFieldName, e.i );
	map< Eref, unsigned int >::iterator j;
	ret.resize( 0 );
	while ( c->good() ) {
		Eref src = c->target();
		j = molMap_.find( src );
		if ( j != molMap_.end() ) {
			ret.push_back( &S_[ j->second ] );
		} else { 
			// Table or other object, not handled.
			// cout << "Error: findTargets: Unable to handle " << src.name() << " as src/target for " << e.name() << "." << msgFieldName << endl;
			;
		}
		c->increment();
	}
	delete c;
	return ( ret.size() > 0 );
}

class ZeroOrder* makeHalfReaction( double k, vector< const double*> v )
{
	class ZeroOrder* ret = 0;
	switch ( v.size() ) {
		case 0:
			ret = new ZeroOrder( k );
			break;
		case 1:
			ret = new FirstOrder( k, v[0] );
			break;
		case 2:
			ret = new SecondOrder( k, v[0], v[1] );
			break;
		default:
			ret = new NOrder( k, v );
			break;
	}
	return ret;
}

void Stoich::fillHalfStoich( const double* baseptr, 
	vector< const double* >& reactant, int sign, unsigned int reacNum )
{
	vector< const double* >::iterator i;
	const double* lastptr = 0;
	int n = 1;
	unsigned int molNum = 0;

	assert( reacNum < v_.size() );
	sort( reactant.begin(), reactant.end() );
	lastptr = reactant.front();
	for (i = reactant.begin() + 1; i != reactant.end(); i++) {
		if ( *i == lastptr ) {
			n++;
		}
		if ( *i != lastptr ) {
			molNum = static_cast< unsigned int >(lastptr - baseptr);
			assert( molNum < nMols_ );
			if ( molNum < nMols_ ) { // If should be redundant here.
				N_.set( molNum, reacNum, sign * n );
				n = 1;
			}
		}
		lastptr = *i;
	}
	molNum = static_cast< unsigned int >(lastptr - baseptr);
	assert( molNum < nMols_ );
	if ( molNum < nMols_ ) { // Should be redundant.
		N_.set( molNum, reacNum, sign * n );
	}
}

void Stoich::fillStoich( 
	const double* baseptr, 
	vector< const double* >& sub, vector< const double* >& prd, 
	unsigned int reacNum )
{
	if ( sub.size() > 0 && prd.size() > 0 ) {
		fillHalfStoich( baseptr, sub, -1 , reacNum );
		fillHalfStoich( baseptr, prd, 1 , reacNum );
	} else {
		/*
		if ( sub.size() == 0 )
			cout << "Warning: Stoich::fillStoich: dangling substrate. Reaction ignored.\n";
		else
			cout << "Warning: Stoich::fillStoich: dangling product. Reaction ignored.\n";
			*/
	}
}

/**
 * Adds the reaction-element e to the solved system.
 */
void Stoich::addReac( Eref stoich, Eref e )
{
	static ZeroOrder* dummyReac = new ZeroOrder( 0.0 );
	vector< const double* > sub;
	vector< const double* > prd;
	class ZeroOrder* freac = 0;
	class ZeroOrder* breac = 0;
	double kf = Reaction::getRawKf( e ); // bypass the solver lookup stuff
	double kb = Reaction::getRawKb( e );
	/*
	double kf;
	double kb;
	bool isOK = get< double >( e, "kf", kf );
	assert ( isOK );
	isOK = get< double >( e, "kb", kb );
	assert ( isOK );
	*/
	assert( !isnan( kf ) );
	assert( !isnan( kb ) );

	if ( findTargets( e, "sub", sub ) ) {
		freac = makeHalfReaction( kf, sub );
	}
	if ( findTargets( e, "prd", prd ) ) {
		breac = makeHalfReaction( kb, prd );
	}
	if ( freac == 0 || breac == 0 ) {
		breac = freac = dummyReac;
		cout << "Stoich::addReac: dummy on " << e.id().path() << "\n";
	// 	assert( 0 );
	// 	return;
	}
	// cout << "addReac " << e.name() << " kf, kb=" << kf << ", " << kb << endl;
	/*
	if ( sub.size() == 0 || prd.size() == 0 ) {
		if ( sub.size() == 0 )
			cout << "Warning: Stoich::fillStoich: dangling substrate on " << e.id().path() << ". Reaction ignored.\n";
		else
			cout << "Warning: Stoich::fillStoich: dangling product on " << e.id().path() << ". Reaction ignored.\n";
		// return;
	}
	*/

#ifdef DO_UNIT_TESTS
	reacMap_[e] = rates_.size();
#endif
	send2< unsigned int, Eref >( stoich, 
			reacConnectionSlot, rates_.size(), e );
	if ( useOneWayReacs_ ) {
		if ( freac ) {
			fillStoich( &S_[0], sub, prd, rates_.size() );
			rates_.push_back( freac );
		}
		if ( breac ) {
			fillStoich( &S_[0], prd, sub, rates_.size() );
			rates_.push_back( breac );
		}
	} else { 
		fillStoich( &S_[0], sub, prd, rates_.size() );
			if ( freac != dummyReac ) {
				rates_.push_back( 
					new BidirectionalReaction( freac, breac )
				);
			} else {
				rates_.push_back( dummyReac );
				// cout << "Stoich::addReac: dummy on " << e.id().path() << "\n";
			}
			/*
			} else if ( freac )  {
				rates_.push_back( freac );
			} else if ( breac ) {
				rates_.push_back( breac );
			} else {
				assert( 0 );
			}
			*/
	}
	++nReacs_;
}

bool Stoich::checkEnz( Eref e,
		vector< const double* >& sub,
		vector< const double* >& prd,
		vector< const double* >& enz,
		vector< const double* >& cplx,
		double& k1, double& k2, double& k3,
		bool isMM
	)
{
	/*
	bool ret;
	ret = get< double >( e, "k1", k1 );
	assert( ret );
	assert( !isnan( k1 ) );
	ret = get< double >( e, "k2", k2 );
	assert( ret );
	assert( !isnan( k2 ) );
	ret = get< double >( e, "k3", k3 );
	assert( ret );
	assert( !isnan( k3 ) );
	*/
	k1 = Enzyme::getK1( e );
	k2 = Enzyme::getK2( e );
	k3 = Enzyme::getK3( e );
	assert( !isnan( k1 ) );
	assert( !isnan( k2 ) );
	assert( !isnan( k3 ) );

	if ( !findTargets( e, "sub", sub ) ) {
		cerr << "Error: Stoich::addEnz( " << e.name() << " ) : Failed to find subs\n";
		return 0;
	}
	if ( !findTargets( e, "enz", enz ) ) {
		cerr << "Error: Stoich::addEnz( " << e.name() << " ) : Failed to find enzyme\n";
		return 0;
	}
	if ( !isMM ) {
		if ( !findTargets( e, "cplx", cplx )  ) {  
			cerr << "Error: Stoich::addEnz( " << e.name() << " ) : Failed to find cplx\n";
			return 0;
		}
	}
	if ( !findTargets( e, "prd", prd ) ) {
		cerr << "Error: Stoich::addEnz( " << e.name() << " ) : Failed to find prds\n";
		return 0;
	}
	return 1;
}

void Stoich::addEnz( Eref stoich, Eref e )
{
	static ZeroOrder* dummyReac = new ZeroOrder( 0.0 );
	vector< const double* > sub;
	vector< const double* > prd;
	vector< const double* > enz;
	vector< const double* > cplx;
	class ZeroOrder* freac = 0;
	class ZeroOrder* breac = 0;
	class ZeroOrder* catreac = 0;
	double k1;
	double k2;
	double k3;
	if ( checkEnz( e, sub, prd, enz, cplx, k1, k2, k3, 0 ) ) {
		sub.push_back( enz[ 0 ] );
		prd.push_back( enz[ 0 ] );
		freac = makeHalfReaction( k1, sub );
		breac = makeHalfReaction( k2, cplx );
		catreac = makeHalfReaction( k3, cplx );
		send2< unsigned int, Eref >(
			stoich, enzConnectionSlot,
			rates_.size(), e );
		if ( useOneWayReacs_ ) {
			fillStoich( &S_[0], sub, cplx, rates_.size() );
			rates_.push_back( freac );
			fillStoich( &S_[0], cplx, sub, rates_.size() );
			rates_.push_back( breac );
			fillStoich( &S_[0], cplx, prd, rates_.size() );
			rates_.push_back( catreac );
		} else { 
			fillStoich( &S_[0], sub, cplx, rates_.size() );
			rates_.push_back( 
				new BidirectionalReaction( freac, breac ) );
			fillStoich( &S_[0], cplx, prd, rates_.size() );
			rates_.push_back( catreac );
		}
		nEnz_++;
	} else {
		if ( useOneWayReacs_ )
			rates_.push_back( dummyReac );
		rates_.push_back( dummyReac );
		rates_.push_back( dummyReac );
	}
}

void Stoich::addMmEnz( Eref stoich, Eref e )
{
	vector< const double* > sub;
	vector< const double* > prd;
	vector< const double* > enz;
	vector< const double* > cplx;
	class ZeroOrder* sublist = 0;
	double k1;
	double k2;
	double k3;
	if ( checkEnz( e, sub, prd, enz, cplx, k1, k2, k3, 1 ) ) {
		double Km = 1.0;
		if ( k1 > EPSILON ) {
			Km = ( k2 + k3 ) / k1;
		} else {
			cerr << "Error: Stoich::addMMEnz: zero k1\n";
			return;
		}
		fillStoich( &S_[0], sub, prd, rates_.size() );
		send2< unsigned int, Eref >(
			stoich, mmEnzConnectionSlot,
			rates_.size(), e );
		if ( sub.size() == 1 ) {
			// cout << "Making MMEnzyme1\n" << flush;
			rates_.push_back( new MMEnzyme1( Km, k3, enz[0], sub[0] ) );
		} else {
			// cout << "Making general MMEnzyme\n" << flush;
			sublist = makeHalfReaction( 1.0, sub );
			rates_.push_back( new MMEnzyme( Km, k3, enz[0], sublist ) );
		}
		nMmEnz_++;
	}
}

void Stoich::addTab( Eref stoich, Eref e )
{
	// nothing to do here. It is handled by keeping the tab as an external,
	// unmanaged object.
	// cout << "Don't yet know how to addTab for " << e->name() << "\n";
}

void Stoich::addRate( Eref stoich, Eref e )
{
	// cout << "Don't yet know how to addRate for " << e->name() << "\n";
}

void Stoich::setupReacSystem( Eref stoich )
{
	// At this point the Stoich has sent out all info to the target
	// objects and the Hubs. It now requests them to do something with it.
	// Since these objects may have to do stuff the stoich doesn't know
	// about, it also sends them the path to work on.
	send1< string >( stoich, completeSetupSlot, path_ );
}

// Update the v_ vector for individual reac velocities.
void Stoich::updateV( )
{
	// Some algorithm to assign the values from the computed rates
	// to the corresponding v_ vector entry
	// for_each( rates_.begin(), rates_.end(), assign);

	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v_.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)();
		assert( !isnan( *( j-1 ) ) );
	}

	// I should use foreach here.
	vector< SumTotal >::const_iterator k;
	for ( k = sumTotals_.begin(); k != sumTotals_.end(); k++ )
		k->sum();
}

void Stoich::updateRates( vector< double>* yprime, double dt  )
{
	updateV();

	// Much scope for optimization here.
	vector< double >::iterator j = yprime->begin();
	assert( yprime->size() >= nVarMols_ );
	for (unsigned int i = 0; i < nVarMols_; i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}

/**
 * Assigns n values for all molecules that have had their slave-enable
 * flag set _after_ the run started. Ugly hack, but convenient for 
 * simulations. Likely to be very few, if any.
 */
void Stoich::updateDynamicBuffers()
{
	// Here we handle dynamic buffering by simply writing over S_.
	// We never see y_ in the rest of the simulation, so can ignore.
	// Main concern is that y_ will go wandering off into nans, or
	// become numerically unhappy and slow things down.
	for ( vector< unsigned int >::const_iterator 
		i = dynamicBuffers_.begin(); i != dynamicBuffers_.end(); ++i )
		S_[ *i ] = Sinit_[ *i ];
}

#ifdef USE_GSL
///////////////////////////////////////////////////
// GSL interface stuff
///////////////////////////////////////////////////

/**
 * This is the function used by GSL to advance the simulation one step.
 * We have a design decision here: to perform the calculations 'in place'
 * on the passed in y and f arrays, or to copy the data over and use
 * the native calculations in the Stoich object. We chose the latter,
 * because memcpy is fast, and the alternative would be to do a huge
 * number of array lookups (currently it is direct pointer references).
 * Someday should benchmark to see how well it works.
 * The derivative array f is used directly by the stoich function
 * updateRates that computes these derivatives, so we do not need to
 * do any memcopies there.
 *
 * Perhaps not by accident, this same functional form is used by CVODE.
 * Should make it easier to eventually use CVODE as a solver too.
 */

// Static function passed in as the stepper for GSL
int Stoich::gslFunc( double t, const double* y, double* yprime, void* s )
{
	return static_cast< Stoich* >( s )->innerGslFunc( t, y, yprime );
}


int Stoich::innerGslFunc( double t, const double* y, double* yprime )
{
	nCall_++;
//	if ( lasty_ != y ) { // should count to see how often this copy happens
		// Copy the y array into the y_ vector.
		memcpy( &S_[0], y, nVarMolsBytes_ );
		lasty_ = y;
		nCopy_++;
//	}

	updateDynamicBuffers();

	updateV();

	// Much scope for optimization here.
	for (unsigned int i = 0; i < nVarMols_; i++) {
		*yprime++ = N_.computeRowRate( i , v_ );
	}
	// cout << t << ": " << y[0] << ", " << y[1] << endl;
	return GSL_SUCCESS;
}

void Stoich::runStats()
{
	cout << "Copy:Call=	" << nCopy_ << ":" << nCall_ << endl;
}


#endif // USE_GSL
