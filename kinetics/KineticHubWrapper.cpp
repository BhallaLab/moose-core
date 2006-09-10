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
#include "KineticHub.h"
#include "KineticHubWrapper.h"


Finfo* KineticHubWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processMolOut", &KineticHubWrapper::getProcessMolSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitMolOut", &KineticHubWrapper::getReinitMolSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"molOut", &KineticHubWrapper::getMolSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"bufOut", &KineticHubWrapper::getBufSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processBufOut", &KineticHubWrapper::getProcessBufSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"reinitBufOut", &KineticHubWrapper::getReinitBufSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"sumTotOut", &KineticHubWrapper::getSumTotSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processSumTotOut", &KineticHubWrapper::getProcessSumTotSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"reinitSumTotOut", &KineticHubWrapper::getReinitSumTotSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processReacOut", &KineticHubWrapper::getProcessReacSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitReacOut", &KineticHubWrapper::getReinitReacSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processEnzOut", &KineticHubWrapper::getProcessEnzSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitEnzOut", &KineticHubWrapper::getReinitEnzSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processMmEnzOut", &KineticHubWrapper::getProcessMmEnzSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitMmEnzOut", &KineticHubWrapper::getReinitMmEnzSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processRateOut", &KineticHubWrapper::getProcessRateSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitRateOut", &KineticHubWrapper::getReinitRateSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"rateOut", &KineticHubWrapper::getRateSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processTabOut", &KineticHubWrapper::getProcessTabSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitTabOut", &KineticHubWrapper::getReinitTabSrc, 
		"", 1 ),
	new SingleSrc0Finfo(
		"updateOut", &KineticHubWrapper::getUpdateSrc, 
		"" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"processIn", &KineticHubWrapper::processFunc,
		&KineticHubWrapper::getProcessInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &KineticHubWrapper::reinitFunc,
		&KineticHubWrapper::getReinitInConn, "" ),
	new Dest3Finfo< int, int, int >(
		"molSizesIn", &KineticHubWrapper::molSizesFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest3Finfo< int, int, int >(
		"rateSizesIn", &KineticHubWrapper::rateSizesFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest3Finfo< vector< double >* , vector< double >* , vector< Element *>*  >(
		"molConnectionsIn", &KineticHubWrapper::molConnectionsFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest2Finfo< vector< RateTerm* >* , int >(
		"rateTermInfoIn", &KineticHubWrapper::rateTermInfoFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest2Finfo< int, Element* >(
		"reacConnectionIn", &KineticHubWrapper::reacConnectionFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest2Finfo< int, Element* >(
		"enzConnectionIn", &KineticHubWrapper::enzConnectionFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest2Finfo< int, Element* >(
		"mmEnzConnectionIn", &KineticHubWrapper::mmEnzConnectionFunc,
		&KineticHubWrapper::getHubConn, "", 1 ),
	new Dest3Finfo< double, double, int >(
		"molIn", &KineticHubWrapper::molFunc,
		&KineticHubWrapper::getMolSolveConn, "", 1 ),
	new Dest3Finfo< double, double, int >(
		"bufIn", &KineticHubWrapper::bufFunc,
		&KineticHubWrapper::getBufSolveConn, "", 1 ),
	new Dest3Finfo< double, double, int >(
		"sumTotIn", &KineticHubWrapper::sumTotFunc,
		&KineticHubWrapper::getSumTotSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"rateIn", &KineticHubWrapper::rateFunc,
		&KineticHubWrapper::getRateSolveConn, "", 1 ),
	new Dest2Finfo< double, double >(
		"reacIn", &KineticHubWrapper::reacFunc,
		&KineticHubWrapper::getReacSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"enzIn", &KineticHubWrapper::enzFunc,
		&KineticHubWrapper::getEnzSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"mmEnzIn", &KineticHubWrapper::mmEnzFunc,
		&KineticHubWrapper::getMmEnzSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"tabIn", &KineticHubWrapper::tabFunc,
		&KineticHubWrapper::getTabSolveConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"molSolve", &KineticHubWrapper::getMolSolveConn,
		"processMolOut, reinitMolOut, molOut, molIn" ),
	new SharedFinfo(
		"bufSolve", &KineticHubWrapper::getBufSolveConn,
		"processBufOut, reinitBufOut, bufOut, bufIn" ),
	new SharedFinfo(
		"sumTotSolve", &KineticHubWrapper::getSumTotSolveConn,
		"processSumTotOut, reinitSumTotOut, sumTotOut, sumTotIn" ),
	new SharedFinfo(
		"reacSolve", &KineticHubWrapper::getReacSolveConn,
		"processReacOut, reinitReacOut, reacIn" ),
	new SharedFinfo(
		"enzSolve", &KineticHubWrapper::getEnzSolveConn,
		"processEnzOut, reinitEnzOut, enzIn" ),
	new SharedFinfo(
		"mmEnzSolve", &KineticHubWrapper::getMmEnzSolveConn,
		"processMmEnzOut, reinitMmEnzOut, mmEnzIn" ),
	new SharedFinfo(
		"tabSolve", &KineticHubWrapper::getTabSolveConn,
		"processTabOut, reinitTabOut, tabIn" ),
	new SharedFinfo(
		"rateSolve", &KineticHubWrapper::getRateSolveConn,
		"processRateOut, reinitRateOut, rateOut, rateIn" ),
	new SharedFinfo(
		"hub", &KineticHubWrapper::getHubConn,
		"molSizesIn, rateSizesIn, rateTermInfoIn, molConnectionsIn, reacConnectionIn, enzConnectionIn, mmEnzConnectionIn" ),
};

const Cinfo KineticHubWrapper::cinfo_(
	"KineticHub",
	"Upinder S. Bhalla, September 2006, NCBS",
	"KineticHub: Object for controlling reaction systems on behalf of the\nStoich object. Interfaces both with the reaction system\n(molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
	"Neutral",
	KineticHubWrapper::fieldArray_,
	sizeof(KineticHubWrapper::fieldArray_)/sizeof(Finfo *),
	&KineticHubWrapper::create
);

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void KineticHubWrapper::molSizesFuncLocal( int nMol, int nBuf, int nSumTot )
{
			vector< unsigned long > segments( 1, 0 );
			nMol_ = segments[0] = nMol;
			Field molSolve( this, "molSolve" );
			molSolve->resize( this, segments );
			nBuf_ = segments[0] = nBuf;
			Field bufSolve( this, "bufSolve" );
			bufSolve->resize( this, segments );
			nSumTot_ = segments[0] = nSumTot;
			Field sumTotSolve( this, "sumTotSolve" );
			sumTotSolve->resize( this, segments );
}
void KineticHubWrapper::rateSizesFuncLocal( int nReac, int nEnz, int nMmEnz )
{
			vector< unsigned long > segments( 1, 0 );
			segments[0] = nReac;
			Field reacSolve( this, "reacSolve" );
			reacSolve->resize( this, segments );
			segments[0] = nEnz;
			Field enzSolve( this, "enzSolve" );
			enzSolve->resize( this, segments );
			segments[0] = nMmEnz;
			Field mmEnzSolve( this, "mmEnzSolve" );
			mmEnzSolve->resize( this, segments );
}
void KineticHubWrapper::molConnectionsFuncLocal( vector< double >*  S, vector< double >*  Sinit, vector< Element *>*  elist )
{
			if ( nMol_ + nBuf_ + nSumTot_ != elist->size() ) {
				cerr << "Error: KineticHub::molConnections: Number of molecules does not match elist size\n";
				return;
			}
			Field molSolve( this, "molSolve" );
			Field sumTotSolve( this, "sumTotSolve" );
			Field bufSolve( this, "bufSolve" );
			S_ = S;
			Sinit_ = Sinit;
			unsigned long i;
			unsigned long max;
			for ( i = 0; i < nMol_; i++ )
				zombify( (*elist)[i], molSolve );
			max = nMol_ + nSumTot_;
			for ( ; i < max; i++ )
				zombify( (*elist)[i], sumTotSolve );
			max = nMol_ + nSumTot_ + nBuf_;
			for ( ; i < max; i++ )
				zombify( (*elist)[i], bufSolve );
}
void KineticHubWrapper::reacConnectionFuncLocal( int rateTermIndex, Element* reac )
{
			Field reacSolve( this, "reacSolve" );
			zombify( reac, reacSolve );
			reacIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::enzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field enzSolve( this, "enzSolve" );
			zombify( enz, enzSolve );
			enzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::mmEnzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field mmEnzSolve( this, "mmEnzSolve" );
			zombify( enz, mmEnzSolve );
			mmEnzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::molFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				molSrc_.sendTo( index, (*S_)[index] );
			} else if ( mode == SOLVER_SET ) {
				(*S_)[index] = n;
				(*Sinit_)[index] = nInit;
			} else if ( mode == SOLVER_REBUILD ) {
				rebuildFlag_ = 1;
			}
}
void KineticHubWrapper::bufFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
			} else if ( mode == SOLVER_SET ) {
				(*S_)[ index + nMol_ + nSumTot_ ] = 
					(*Sinit_)[ index + nMol_ + nSumTot_ ] = nInit;
			}
}
void KineticHubWrapper::sumTotFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				sumTotSrc_.sendTo( index, (*S_)[index + nMol_ ] );
			}
}
void KineticHubWrapper::reacFuncLocal( double kf, double kb, long index )
{
			unsigned long i = reacIndex_[ index ];
			if ( i >= 0 && i < rates_->size() - useOneWayReacs_ ) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( kf, 0 );
					( *rates_ )[i + 1]->setRates( kb, 0 );
				} else {
					( *rates_ )[i]->setRates( kf, kb );
				}
			}
}
void KineticHubWrapper::enzFuncLocal( double k1, double k2, double k3, long index )
{
			unsigned int i = enzIndex_[ index ];
			if ( i < rates_->size() - useOneWayReacs_ - 1) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( k1, 0 );
					( *rates_ )[i + 1]->setRates( k2, 0 );
					( *rates_ )[i + 2]->setRates( k3, 0 );
				} else {
					( *rates_ )[i]->setRates( k1, k2 );
					( *rates_ )[i + 1]->setRates( k3, 0 );
				}
			}
}
void KineticHubWrapper::mmEnzFuncLocal( double k1, double k2, double k3, long index )
{
			double Km = ( k2 + k3 ) / k1 ;
			unsigned int i = mmEnzIndex_[ index ];
			if ( i >= 0 && i < rates_->size() )
				( *rates_ )[i]->setRates( Km, k3 );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* hubConnKineticHubLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KineticHubWrapper, hubConn_ );
	return reinterpret_cast< KineticHubWrapper* >( ( unsigned long )c - OFFSET );
}

Element* updateOutConnKineticHubLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KineticHubWrapper, updateOutConn_ );
	return reinterpret_cast< KineticHubWrapper* >( ( unsigned long )c - OFFSET );
}

Element* processInConnKineticHubLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KineticHubWrapper, processInConn_ );
	return reinterpret_cast< KineticHubWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reinitInConnKineticHubLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KineticHubWrapper, reinitInConn_ );
	return reinterpret_cast< KineticHubWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void KineticHubWrapper::zombify( Element* e, Field& solveSrc )
{
	Field f( e, "process" );
	if ( !f.dropAll() ) {
		cerr << "Error: Failed to delete process message into " <<
			e->path() << "\n";
	}
	Field ms( e, "solve" );
	if ( !solveSrc.add( ms ) ) {
		cerr << "Error: Failed to add solve message from solver " <<
			solveSrc.path() << " to zombie " << e->path() << "\n";
	}
}
