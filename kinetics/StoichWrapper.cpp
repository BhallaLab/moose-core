/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "Stoich.h"
#include "StoichWrapper.h"

const double StoichWrapper::EPSILON = 1.0e-6;

Finfo* StoichWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ReadOnlyValueFinfo< int >(
		"nMols", &StoichWrapper::getNMols, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nVarMols", &StoichWrapper::getNVarMols, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nSumTot", &StoichWrapper::getNSumTot, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nBuffered", &StoichWrapper::getNBuffered, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nReacs", &StoichWrapper::getNReacs, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nEnz", &StoichWrapper::getNEnz, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nMmEnz", &StoichWrapper::getNMmEnz, "int" ),
	new ReadOnlyValueFinfo< int >(
		"nExternalRates", &StoichWrapper::getNExternalRates, "int" ),
	new ValueFinfo< int >(
		"useOneWayReacs", &StoichWrapper::getUseOneWayReacs, 
		&StoichWrapper::setUseOneWayReacs, "int" ),
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"path", &StoichWrapper::getPath, 
		&StoichWrapper::setPath, "string" ),
	new ReadOnlyValueFinfo< int >(
		"rateVectorSize", &StoichWrapper::getRateVectorSize, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
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
	new SingleSrc3Finfo< vector< double >* , vector< double >* , vector< Element *>*  >(
		"molConnectionsOut", &StoichWrapper::getMolConnectionsSrc, 
		"", 1 ),
	new SingleSrc2Finfo< int, Element* >(
		"reacConnectionOut", &StoichWrapper::getReacConnectionSrc, 
		"", 1 ),
	new SingleSrc2Finfo< int, Element* >(
		"enzConnectionOut", &StoichWrapper::getEnzConnectionSrc, 
		"", 1 ),
	new SingleSrc2Finfo< int, Element* >(
		"mmEnzConnectionOut", &StoichWrapper::getMmEnzConnectionSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &StoichWrapper::reinitFunc,
		&StoichWrapper::getIntegrateConn, "allocateOut", 1 ),
	new Dest2Finfo< vector< double >* , double >(
		"integrateIn", &StoichWrapper::integrateFunc,
		&StoichWrapper::getIntegrateConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"integrate", &StoichWrapper::getIntegrateConn,
		"integrateIn, allocateOut, reinitIn" ),
	new SharedFinfo(
		"hub", &StoichWrapper::getHubConn,
		"molSizesOut, rateSizesOut, rateTermInfoOut, molConnectionsOut, reacConnectionOut, enzConnectionOut, mmEnzConnectionOut" ),
};

const Cinfo StoichWrapper::cinfo_(
	"Stoich",
	"Upinder S. Bhalla, April 2006, NCBS",
	"Stoich: Sets up stoichiometry matrix based calculations from a\nwildcard path for the reaction system.\nKnows how to compute derivatives for most common\nthings, also knows how to handle special cases where the\nobject will have to do its own computation. Generates a\nstoichiometry matrix, which is useful for lots of other\noperations as well.",
	"Neutral",
	StoichWrapper::fieldArray_,
	sizeof(StoichWrapper::fieldArray_)/sizeof(Finfo *),
	&StoichWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

string StoichWrapper::localGetPath() const
{
			return path_;
}
void StoichWrapper::localSetPath( string value ) {
			setPathLocal( value );
}
int StoichWrapper::localGetRateVectorSize() const
{
			return rates_.size();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* integrateConnStoichLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( StoichWrapper, integrateConn_ );
	return reinterpret_cast< StoichWrapper* >( ( unsigned long )c - OFFSET );
}

Element* hubConnStoichLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( StoichWrapper, hubConn_ );
	return reinterpret_cast< StoichWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
unsigned int countRates( Element* e, bool useOneWayReacs )
{
	if ( e-> cinfo()->name() == "Reaction" ) {
		if ( useOneWayReacs)
			return 2;
		else
			return 1;
	}
	if ( e->cinfo()->name() == "Enzyme" ) {
		int mode = 0;
		if ( Ftype1< int >::get( e, "mode", mode ) ) {
			if ( mode == 0 ) { 
				if ( useOneWayReacs )
					return 3;
				else
					return 2;
			} else { 
				return 1;
			}
		}
	}
	return 0;
}
void StoichWrapper::setPathLocal( const string& value )
{
	path_ = value;
	vector< Element* > ret;
	vector< Element* >::iterator i;
	Element::startFind( path_, ret );
	vector< Element* > varMolVec;
	vector< Element* > bufVec;
	vector< Element* > sumTotVec;
	int mode;
	unsigned int numRates = 0;
	const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->cinfo() == molCinfo ) {
			if ( Ftype1< int >::get( *i, "mode", mode ) ) {
				if ( mode == 0 ) {
					varMolVec.push_back( *i );
				} else if ( mode == 4 ) {
					bufVec.push_back( *i );
				} else {
					sumTotVec.push_back( *i );
				}
			}
		} else {
			numRates += countRates( *i, useOneWayReacs_ );
		}
	}
	setupMols( varMolVec, bufVec, sumTotVec );
	N_.setSize( varMolVec.size() , numRates );
	v_.resize( numRates, 0.0 );
	rateTermInfoSrc_.send( &rates_, useOneWayReacs_ );
	const Cinfo* reacCinfo = Cinfo::find( "Reaction" );
	const Cinfo* enzCinfo = Cinfo::find( "Enzyme" );
	int nReac = 0;
	int nEnz = 0;
	int nMmEnz = 0;
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->cinfo() == reacCinfo ) {
			nReac++;
		} else if ( ( *i )->cinfo() == enzCinfo ) {
			int mode = 0;
			if ( Ftype1< int >::get( *i, "mode", mode ) ) {
				if ( mode == 0 )
					nEnz++;
				else
					nMmEnz++;
			}
		}
	}
	rateSizesSrc_.send( nReac, nEnz, nMmEnz );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( ( *i )->cinfo() == reacCinfo ) {
			addReac( *i );
		} else if ( ( *i )->cinfo() == enzCinfo ) {
			int mode = 0;
			if ( Ftype1< int >::get( *i, "mode", mode ) ) {
				if ( mode == 0 )
					addEnz( *i );
				else
					addMmEnz( *i );
			} else {
				cerr << "Error: StoichWrapper::innerSetPath: Error getting mode for enz\n" << (*i)->path() << "\n";
			}
		} else if ( ( *i )->cinfo()->name() == "table" ) {
			addTab( *i );
		} else {
			addRate( *i );
		}
	}
	setupReacSystem();
}
void StoichWrapper::setupMols(
	vector< Element* >& varMolVec,
	vector< Element* >& bufVec,
	vector< Element* >& sumTotVec
	)
{
	Field nInitField = Cinfo::find( "Molecule" )->field( "nInit" );
	vector< Element* >::iterator i;
	vector< Element* >elist;
	int j = 0;
	double nInit;
	nVarMols_ = varMolVec.size();
	nSumTot_  = sumTotVec.size();
	nBuffered_ = bufVec.size();
	nMols_ = nVarMols_ + nSumTot_ + nBuffered_;
	S_.resize( nMols_ );
	Sinit_.resize( nMols_ );
	for ( i = varMolVec.begin(); i != varMolVec.end(); i++ ) {
		Ftype1< double >::get( *i, nInitField.getFinfo(), nInit );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = sumTotVec.begin(); i != sumTotVec.end(); i++ ) {
		Ftype1< double >::get( *i, nInitField.getFinfo(), nInit );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = bufVec.begin(); i != bufVec.end(); i++ ) {
		Ftype1< double >::get( *i, nInitField.getFinfo(), nInit );
		Sinit_[j] = nInit;
		molMap_[ *i ] = j++;
		elist.push_back( *i );
	}
	for ( i = sumTotVec.begin(); i != sumTotVec.end(); i++ ) {
		addSumTot( *i );
	}
	molSizesSrc_.send( nVarMols_, nBuffered_, nSumTot_ );
	molConnectionsSrc_.send( &S_, &Sinit_, &elist );
}
void StoichWrapper::addSumTot( Element* e )
{
	vector< Field > srclist;
	vector< Field >::iterator i;
	Field srcField = e->field( "sumTotalIn" );
	srcField.dest( srclist );
	for (i = srclist.begin() ; i != srclist.end(); i++ ) {
	}
}
unsigned int StoichWrapper::findReactants( 
	Element* e, const string& msgFieldName, 
	vector< const double* >& ret )
{
	vector< Field > srclist;
	vector< Field >::iterator i;
	ret.resize( 0 );
	map< const Element*, int >::iterator j;
	Field srcField = e->field( msgFieldName );
	srcField.src( srclist );
	for (i = srclist.begin() ; i != srclist.end(); i++ ) {
		Element* src = i->getElement();
		j = molMap_.find( src );
		if ( j != molMap_.end() ) {
			ret.push_back( & S_[ j->second ] );
		} else {
			cerr << "Error: Unable to locate " << src->path() <<
				" as reactant for " << e->path();
			return 0;
		}
	}
	return ret.size();
}
unsigned int StoichWrapper::findProducts( 
	Element* e, const string& msgFieldName, 
	vector< const double* >& ret )
{
	vector< Field > prdlist;
	vector< Field >::iterator i;
	ret.resize( 0 );
	map< const Element*, int >::iterator j;
	Field prdField = e->field( msgFieldName );
	prdField.dest( prdlist );
	for (i = prdlist.begin() ; i != prdlist.end(); i++ ) {
		Element* prd = i->getElement();
		j = molMap_.find( prd );
		if ( j != molMap_.end() ) {
			ret.push_back( & S_[ j->second ] );
		} else {
			cerr << "Error: Unable to locate " << prd->path() <<
				" as reactant for " << e->path();
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
void StoichWrapper::fillHalfStoich( const double* baseptr, 
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
void StoichWrapper::fillStoich( 
	const double* baseptr, 
	vector< const double* >& sub, vector< const double* >& prd, 
	int reacNum )
{
	fillHalfStoich( baseptr, sub, 1 , reacNum );
	fillHalfStoich( baseptr, prd, -1 , reacNum );
}
void StoichWrapper::addReac( Element* e )
{
	vector< const double* > sub;
	vector< const double* > prd;
	class ZeroOrder* freac = 0;
	class ZeroOrder* breac = 0;
	double kf;
	double kb;
	if ( Ftype1< double >::get( e, "kf", kf ) &&
		 Ftype1< double >::get( e, "kb", kb )
	) {
		if ( findReactants( e, "subIn", sub ) > 0 ) {
			freac = makeHalfReaction( kf, sub );
		}
		if ( findReactants( e, "prdIn", prd ) > 0 ) {
			breac = makeHalfReaction( kb, prd );
		}
		reacConnectionSrc_.send( rates_.size(), e );
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
			if ( freac && breac ) {
				rates_.push_back( 
					new BidirectionalReaction( freac, breac ) );
			} else if ( freac )  {
				rates_.push_back( freac );
			} else if ( breac ) {
				rates_.push_back( breac );
			}
		}
		++nReacs_;
	}
}
bool StoichWrapper::checkEnz( Element* e,
		vector< const double* >& sub,
		vector< const double* >& prd,
		vector< const double* >& enz,
		vector< const double* >& cplx,
		double& k1, double& k2, double& k3,
		bool isMM
	)
{
	if ( !( 
		Ftype1< double >::get( e, "k1", k1 ) &&
		Ftype1< double >::get( e, "k2", k2 ) &&
		Ftype1< double >::get( e, "k3", k3 )
		)
	) {
		cerr << "Error: StoichWrapper:: Failed to find rates\n";
		return 0;
	}
	if ( findReactants( e, "subIn", sub ) < 1 ) {
		cerr << "Error: StoichWrapper::addEnz: Failed to find subs\n";
		return 0;
	}
	if ( findReactants( e, "enzIn", enz ) != 1 ) {
		cerr << "Error: StoichWrapper::addEnz: Failed to find enzyme\n";
		return 0;
	}
	if ( !isMM ) {
		if ( findReactants( e, "cplxIn", cplx ) != 1 ) {  
			cerr << "Error: StoichWrapper::addEnz: Failed to find cplx\n";
			return 0;
		}
	}
	if ( findProducts( e, "prdOut", prd ) < 1 ) {
		cerr << "Error: StoichWrapper::addEnz: Failed to find prds\n";
		return 0;
	}
	return 1;
}
void StoichWrapper::addEnz( Element* e )
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
		enzConnectionSrc_.send( rates_.size(), e );
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
void StoichWrapper::addMmEnz( Element* e )
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
		mmEnzConnectionSrc_.send( rates_.size(), e );
		rates_.push_back( new MMEnzyme( Km, k3, enz[0], sublist ) );
		nMmEnz_++;
	}
}
void StoichWrapper::addTab( Element* e )
{
}
void StoichWrapper::addRate( Element* e )
{
}
void StoichWrapper::setupReacSystem()
{
}
