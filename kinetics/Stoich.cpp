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
#include "SparseMatrix.h"
#include "Stoich.h"

const double Stoich::EPSILON = 1.0e-6;

const Cinfo* initStoichCinfo()
{
	/*
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Stoich::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Stoich::reinitFunc ) ),
	};
	*/

	static Finfo* hubShared[] =
	{
		new SrcFinfo( "rateTermInfoSrc", 
			Ftype2< vector< RateTerm* >*, bool >::global()
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
				vector< Element *>*  
				>::global() 
		),
		new SrcFinfo( "reacConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
		new SrcFinfo( "enzConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
		new SrcFinfo( "mmEnzConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
	};

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
		new ValueFinfo( "rateVectorSize", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Stoich::getRateVectorSize ), 
			&dummyFunc
		),
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		
		/* Moved over to hubShared
		new SrcFinfo( "rateTermInfoSrc", 
			Ftype2< vector< RateTerm* >*, bool >::global()
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
				vector< Element *>*  
				>::global() 
		),
		new SrcFinfo( "reacConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
		new SrcFinfo( "enzConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
		new SrcFinfo( "mmEnzConnectionSrc",
			Ftype2< unsigned int, Element* >::global()
		),
		*/
		/*
	new SingleSrc1Finfo< vector< double >*  >(
		"allocateOut", &StoichWrapper::getAllocateSrc, 
		"reinitIn", 1 ),
	new SingleSrc3Finfo< int, int, int >(
		"molSizesOut", &StoichWrapper::getMolSizesSrc, 
		"", 1 ),
	new SingleSrc3Finfo< int, int, int >(
		"rateSizesOut", &StoichWrapper::getRateSizesSrc, 
		"", 1 ),
	new SingleSrc2Finfo< vector< RateTerm* >*, int >(
		"rateTermInfoOut", &StoichWrapper::getRateTermInfoSrc, 
		"", 1 ),
		*/
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		/*
		new Dest0Finfo(
			"reinitIn", &StoichWrapper::reinitFunc,
			&StoichWrapper::getIntegrateConn, "allocateOut", 1 ),
		new Dest2Finfo< vector< double >* , double >(
			"integrateIn", &StoichWrapper::integrateFunc,
			&StoichWrapper::getIntegrateConn, "", 1 ),
		
		*/
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "hub", hubShared, 
				sizeof( hubShared )/ sizeof( Finfo* ) ),

		/*
		new SharedFinfo(
			"integrate", &StoichWrapper::getIntegrateConn,
			"integrateIn, allocateOut, reinitIn" ),
		new SharedFinfo(
			"hub", &StoichWrapper::getHubConn,
			"molSizesOut, rateSizesOut, rateTermInfoOut, molConnectionsOut, reacConnectionOut, enzConnectionOut, mmEnzConnectionOut" ),
		*/
	};

	static Cinfo stoichCinfo(
		"Stoich",
		"Upinder S. Bhalla, 2007, NCBS",
		"Stoich: Sets up stoichiometry matrix based calculations from a\nwildcard path for the reaction system.\nKnows how to compute derivatives for most common\nthings, also knows how to handle special cases where the\nobject will have to do its own computation. Generates a\nstoichiometry matrix, which is useful for lots of other\noperations as well.",
		initNeutralCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos )/sizeof(Finfo *),
		ValueFtype1< Stoich >::global()
	);

	return &stoichCinfo;
}

static const Cinfo* stoichCinfo = initStoichCinfo();

static const unsigned int rateTermInfoSlot =
	initStoichCinfo()->getSlotIndex( "rateTermInfoSrc" );
static const unsigned int rateSizeSlot =
	initStoichCinfo()->getSlotIndex( "rateSizeSrc" );
static const unsigned int molSizeSlot =
	initStoichCinfo()->getSlotIndex( "molSizeSrc" );
static const unsigned int molConnectionSlot =
	initStoichCinfo()->getSlotIndex( "molConnectionSrc" );
static const unsigned int reacConnectionSlot =
	initStoichCinfo()->getSlotIndex( "reacConnectionSrc" );
static const unsigned int enzConnectionSlot =
	initStoichCinfo()->getSlotIndex( "enzConnectionSrc" );
static const unsigned int mmEnzConnectionSlot =
	initStoichCinfo()->getSlotIndex( "mmEnzConnectionSrc" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int Stoich::getNmols( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nMols_;
}

unsigned int Stoich::getNvarMols( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nVarMols_;
}

unsigned int Stoich::getNsumTot( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nSumTot_;
}

unsigned int Stoich::getNbuffered( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nBuffered_;
}

unsigned int Stoich::getNreacs( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nReacs_;
}

unsigned int Stoich::getNenz( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nEnz_;
}

unsigned int Stoich::getNmmEnz( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nMmEnz_;
}

unsigned int Stoich::getNexternalRates( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->nExternalRates_;
}
void Stoich::setUseOneWayReacs( const Conn& c, int value ) {
	static_cast< Stoich* >( c.data() )->useOneWayReacs_ = value;
}

bool Stoich::getUseOneWayReacs( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->
		useOneWayReacs_;
}
string Stoich::getPath( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->path_;
}

void Stoich::setPath( const Conn& c, string value ) {
	Element* e = c.targetElement();
	static_cast< Stoich* >( e->data() )->localSetPath( e, value);
}

unsigned int Stoich::getRateVectorSize( const Element* e ) {
	return static_cast< const Stoich* >( e->data() )->rates_.size();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

unsigned int countRates( Element* e, bool useOneWayReacs )
{
	if ( e->className() == "Reaction" ) {
		if ( useOneWayReacs)
			return 2;
		else
			return 1;
	}
	if ( e->className() == "Enzyme" ) {
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
void Stoich::localSetPath( Element* stoich, const string& value )
{
	path_ = value;
	vector< Element* > ret;
	vector< Element* >::iterator i;
	wildcardFind( path_, ret );
	vector< Element* > varMolVec;
	vector< Element* > bufVec;
	vector< Element* > sumTotVec;
	int mode;
	bool isOK;
	unsigned int numRates = 0;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->className() == "Molecule" ) {
			isOK = get< int >( *i, "mode", mode );
			assert( isOK );
			if ( mode == 0 ) {
				varMolVec.push_back( *i );
			} else if ( mode == 4 ) {
				bufVec.push_back( *i );
			} else {
				sumTotVec.push_back( *i );
			}
		} else {
			numRates += countRates( *i, useOneWayReacs_ );
		}
	}
	setupMols( stoich, varMolVec, bufVec, sumTotVec );
	N_.setSize( varMolVec.size() , numRates );
	v_.resize( numRates, 0.0 );
	send2< vector< RateTerm* >*, bool >( 
		stoich, rateTermInfoSlot, &rates_, useOneWayReacs_ );
	// rateTermInfoSrc_.send( &rates_, useOneWayReacs_ );
	int nReac = 0;
	int nEnz = 0;
	int nMmEnz = 0;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->className() == "Reaction" ) {
			nReac++;
		} else if ( ( *i )->className() == "Enzyme" ) {
			bool enzmode = 0;
			isOK = get< bool >( *i, "mode", enzmode );
			assert( isOK );
			if ( enzmode == 0 )
				nEnz++;
			else
				nMmEnz++;
		}
	}
	send3< unsigned int, unsigned int, unsigned int >(
			stoich, rateSizeSlot, nReac, nEnz, nMmEnz );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->className() == "Reaction" ) {
			addReac( stoich, *i );
		} else if ( ( *i )->className() == "Enzyme" ) {
			bool enzmode = 0;
			isOK = get< bool >( *i, "mode", enzmode );
			assert( isOK );
			if ( mode == 0 )
				addEnz( stoich, *i );
			else
				addMmEnz( stoich, *i );
		} else if ( ( *i )->className() == "Table" ) {
			addTab( stoich, *i );
		} else if ( ( *i )->className() != "Molecule" ) {
			addRate( stoich, *i );
		}
	}
	setupReacSystem();
}


void Stoich::setupMols(
	Element* e,
	vector< Element* >& varMolVec,
	vector< Element* >& bufVec,
	vector< Element* >& sumTotVec
	)
{
	const Finfo* nInitFinfo = Cinfo::find( "Molecule" )->
		findFinfo( "nInit" );
	// Field nInitField = Cinfo::find( "Molecule" )->field( "nInit" );
	vector< Element* >::iterator i;
	vector< Element* >elist;
	unsigned int j = 0;
	double nInit;
	nVarMols_ = varMolVec.size();
	nSumTot_  = sumTotVec.size();
	nBuffered_ = bufVec.size();
	nMols_ = nVarMols_ + nSumTot_ + nBuffered_;
	S_.resize( nMols_ );
	Sinit_.resize( nMols_ );
	for ( i = varMolVec.begin(); i != varMolVec.end(); i++ ) {
		get< double >( *i, nInitFinfo, nInit );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = sumTotVec.begin(); i != sumTotVec.end(); i++ ) {
		get< double >( *i, nInitFinfo, nInit );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = bufVec.begin(); i != bufVec.end(); i++ ) {
		get< double >( *i, nInitFinfo, nInit );
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
	send3< vector< double >* , vector< double >* , vector< Element *>*  >(
		e, molConnectionSlot, &S_, &Sinit_, &elist );
}

void Stoich::addSumTot( Element* e )
{
	/*
	static const Finfo* sumTotFinfo = Cinfo::find( "Molecule" )->
		findFinfo( "sumTotal" );
	vector< Element* >::iterator i;
	srcField.dest( srclist );
	for (i = srclist.begin() ; i != srclist.end(); i++ ) {
	}
	*/
}

/**
 * Looks for all substrates of reaction or enzyme
 */
unsigned int Stoich::findReactants( 
	Element* e, const string& msgFieldName, 
	vector< const double* >& ret )
{
	const Finfo* srcFinfo = e->findFinfo( msgFieldName );
	assert( srcFinfo != 0 );
	vector< Conn > srcList;
	vector< Conn >::iterator i;
	ret.resize( 0 );
	map< const Element*, unsigned int >::iterator j;
	// Fill in the list of incoming messages.
	srcFinfo->outgoingConns( e, srcList );
	for (i = srcList.begin() ; i != srcList.end(); i++ ) {
		Element* src = i->targetElement();
		j = molMap_.find( src );
		if ( j != molMap_.end() ) {
			ret.push_back( & S_[ j->second ] );
		} else {
			cerr << "Error: Unable to locate " << 
				src->name() <<
				" as reactant for " << e->name();
			return 0;
		}
	}
	return ret.size();
}

unsigned int Stoich::findProducts( 
	Element* e, const string& msgFieldName, 
	vector< const double* >& ret )
{
	const Finfo* prdFinfo = e->findFinfo( msgFieldName );
	assert( prdFinfo != 0 );
	vector< Conn > prdList;
	vector< Conn >::iterator i;
	ret.resize( 0 );
	map< const Element*, unsigned int >::iterator j;
	// Fill in the list of incoming messages.
	prdFinfo->incomingConns( e, prdList );
	for (i = prdList.begin() ; i != prdList.end(); i++ ) {
		Element* prd = i->targetElement();
		j = molMap_.find( prd );
		if ( j != molMap_.end() ) {
			ret.push_back( & S_[ j->second ] );
		} else {
			cerr << "Error: Unable to locate " << prd->name() <<
				" as product for " << e->name();
			return 0;
		}
	}
	return ret.size();
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
	vector< const double* >& reactant, int sign, int reacNum )
{
	vector< const double* >::iterator i;
	const double* lastptr = 0;
	int n = 1;
	sort( reactant.begin(), reactant.end() );
	lastptr = reactant.front();
	for (i = reactant.begin() + 1; i != reactant.end(); i++) {
		if ( *i == lastptr ) {
			n++;
		}
		if ( *i != lastptr ) {
			N_.set( lastptr - baseptr, reacNum, sign * n );
			n = 1;
		}
		lastptr = *i;
	}
	N_.set( lastptr - baseptr, reacNum, sign * n );
}

void Stoich::fillStoich( 
	const double* baseptr, 
	vector< const double* >& sub, vector< const double* >& prd, 
	int reacNum )
{
	fillHalfStoich( baseptr, sub, -1 , reacNum );
	fillHalfStoich( baseptr, prd, 1 , reacNum );
}

/**
 * Adds the reaction-element e to the solved system.
 */
void Stoich::addReac( Element* stoich, Element* e )
{
	vector< const double* > sub;
	vector< const double* > prd;
	class ZeroOrder* freac = 0;
	class ZeroOrder* breac = 0;
	double kf;
	double kb;
	bool isOK = get< double >( e, "kf", kf );
	assert ( isOK );
	isOK = get< double >( e, "kb", kb );
	assert ( isOK );

	if ( findReactants( e, "sub", sub ) > 0 ) {
		freac = makeHalfReaction( kf, sub );
	}
	if ( findReactants( e, "prd", prd ) > 0 ) {
		breac = makeHalfReaction( kb, prd );
	}

#ifdef DO_UNIT_TESTS
	reacMap_[e] = rates_.size();
#endif
	send2< unsigned int, Element* >( stoich, 
			reacConnectionSlot, rates_.size(), e );
	if ( useOneWayReacs_ ) {
		if ( freac ) {
			fillStoich( &S_[0], sub, prd, rates_.size());
			rates_.push_back( freac );
		}
		if ( breac ) {
			fillStoich( &S_[0], prd, sub, rates_.size());
			rates_.push_back( breac );
		}
	} else { 
		fillStoich( &S_[0], sub, prd, rates_.size() );
		if ( freac && breac ) {
			rates_.push_back( 
				new BidirectionalReaction(
				       	freac, breac ) );
		} else if ( freac )  {
			rates_.push_back( freac );
		} else if ( breac ) {
			rates_.push_back( breac );
		}
	}
	++nReacs_;
}

bool Stoich::checkEnz( Element* e,
		vector< const double* >& sub,
		vector< const double* >& prd,
		vector< const double* >& enz,
		vector< const double* >& cplx,
		double& k1, double& k2, double& k3,
		bool isMM
	)
{
	bool ret;
	ret = get< double >( e, "k1", k1 );
	assert( ret );
	ret = get< double >( e, "k2", k2 );
	assert( ret );
	ret = get< double >( e, "k3", k3 );
	assert( ret );

	if ( findReactants( e, "sub", sub ) < 1 ) {
		cerr << "Error: Stoich::addEnz: Failed to find subs\n";
		return 0;
	}
	if ( findReactants( e, "enz", enz ) != 1 ) {
		cerr << "Error: Stoich::addEnz: Failed to find enzyme\n";
		return 0;
	}
	if ( !isMM ) {
		if ( findReactants( e, "cplx", cplx ) != 1 ) {  
			cerr << "Error: Stoich::addEnz: Failed to find cplx\n";
			return 0;
		}
	}
	if ( findProducts( e, "prdOut", prd ) < 1 ) {
		cerr << "Error: Stoich::addEnz: Failed to find prds\n";
		return 0;
	}
	return 1;
}

void Stoich::addEnz( Element* stoich, Element* e )
{
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
		send2< unsigned int, Element* >(
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
	}
}

void Stoich::addMmEnz( Element* stoich, Element* e )
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
			cerr << "Error: StoichWrapper::addMMEnz: zero k1\n";
			return;
		}
		fillStoich( &S_[0], sub, prd, rates_.size() );
		sublist = makeHalfReaction( 1.0, sub );
		send2< unsigned int, Element* >(
			stoich, mmEnzConnectionSlot,
			rates_.size(), e );
		rates_.push_back( new MMEnzyme( Km, k3, enz[0], sublist ) );
		nMmEnz_++;
	}
}

void Stoich::addTab( Element* stoich, Element* e )
{
	cout << "Don't yet know how to addTab for " << e->name() << "\n";
}

void Stoich::addRate( Element* stoich, Element* e )
{
	cout << "Don't yet know how to addRate for " << e->name() << "\n";
}

void Stoich::setupReacSystem()
{
	cout << "Don't yet know how to setupReacSystem\n";
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
	}
}

void Stoich::updateRates( vector< double>* yprime, double dt  )
{
	updateV();

	// Much scope for optimization here.
	vector< double >::iterator j = yprime->begin();
	for (unsigned int i = 0; i < N_.nRows(); i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}

