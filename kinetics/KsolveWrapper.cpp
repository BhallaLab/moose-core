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
#include "Ksolve.h"
#include "KsolveWrapper.h"


Finfo* KsolveWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"path", &KsolveWrapper::getPath, 
		&KsolveWrapper::setPath, "string" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &KsolveWrapper::getProcessSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitOut", &KsolveWrapper::getReinitSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"molOut", &KsolveWrapper::getMolSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"bufOut", &KsolveWrapper::getBufSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"sumTotOut", &KsolveWrapper::getSumTotSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processReacOut", &KsolveWrapper::getProcessReacSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitReacOut", &KsolveWrapper::getReinitReacSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processEnzOut", &KsolveWrapper::getProcessEnzSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitEnzOut", &KsolveWrapper::getReinitEnzSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processMmEnzOut", &KsolveWrapper::getProcessMmEnzSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitMmEnzOut", &KsolveWrapper::getReinitMmEnzSrc, 
		"", 1 ),
	new NSrc1Finfo< ProcInfo >(
		"processRateOut", &KsolveWrapper::getProcessRateSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reinitRateOut", &KsolveWrapper::getReinitRateSrc, 
		"", 1 ),
	new NSrc1Finfo< double >(
		"rateOut", &KsolveWrapper::getRateSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"processIn", &KsolveWrapper::processFunc,
		&KsolveWrapper::getProcessInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &KsolveWrapper::reinitFunc,
		&KsolveWrapper::getReinitInConn, "" ),
	new Dest3Finfo< double, double, int >(
		"molIn", &KsolveWrapper::molFunc,
		&KsolveWrapper::getMolSolveConn, "", 1 ),
	new Dest3Finfo< double, double, int >(
		"bufMolIn", &KsolveWrapper::bufMolFunc,
		&KsolveWrapper::getMolSolveConn, "", 1 ),
	new Dest3Finfo< double, double, int >(
		"sumTotMolIn", &KsolveWrapper::sumTotMolFunc,
		&KsolveWrapper::getMolSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"rateIn", &KsolveWrapper::rateFunc,
		&KsolveWrapper::getRateSolveConn, "", 1 ),
	new Dest2Finfo< double, double >(
		"reacIn", &KsolveWrapper::reacFunc,
		&KsolveWrapper::getReacSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"enzIn", &KsolveWrapper::enzFunc,
		&KsolveWrapper::getEnzSolveConn, "", 1 ),
	new Dest3Finfo< double, double, double >(
		"mmEnzIn", &KsolveWrapper::mmEnzFunc,
		&KsolveWrapper::getMmEnzSolveConn, "", 1 ),
	new Dest1Finfo< double >(
		"tabIn", &KsolveWrapper::tabFunc,
		&KsolveWrapper::getTabSolveConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"molSolve", &KsolveWrapper::getMolSolveConn,
		"processOut, reinitOut, molOut, molIn" ),
	new SharedFinfo(
		"bufSolve", &KsolveWrapper::getMolSolveConn,
		"processOut, reinitOut, bufOut, bufMolIn" ),
	new SharedFinfo(
		"sumTotSolve", &KsolveWrapper::getMolSolveConn,
		"processOut, reinitOut, sumTotOut, sumTotMolIn" ),
	new SharedFinfo(
		"reacSolve", &KsolveWrapper::getReacSolveConn,
		"processReacOut, reinitReacOut, reacIn" ),
	new SharedFinfo(
		"enzSolve", &KsolveWrapper::getEnzSolveConn,
		"processEnzOut, reinitEnzOut, enzIn" ),
	new SharedFinfo(
		"mmEnzSolve", &KsolveWrapper::getMmEnzSolveConn,
		"processMmEnzOut, reinitMmEnzOut, mmEnzIn" ),
	new SharedFinfo(
		"tabSolve", &KsolveWrapper::getTabSolveConn,
		"processTabOut, reinitTabOut, tabIn" ),
	new SharedFinfo(
		"rateSolve", &KsolveWrapper::getRateSolveConn,
		"processRateOut, reinitRateOut, rateOut, rateIn" ),
};

const Cinfo KsolveWrapper::cinfo_(
	"Ksolve",
	"Upinder S. Bhalla, June 2006, NCBS",
	"Ksolve: Wrapper object for zombifying reaction systems. Interfaces\nboth with the reaction system (molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
	"Neutral",
	KsolveWrapper::fieldArray_,
	sizeof(KsolveWrapper::fieldArray_)/sizeof(Finfo *),
	&KsolveWrapper::create
);

///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

string KsolveWrapper::localGetPath() const
{
			return path_;
}
void KsolveWrapper::localSetPath( const string& value ) {
			path_ = value;
			Ksolve::setPath( value, this );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void KsolveWrapper::molFuncLocal( double n, double nInit, int mode, long index )
{
			cout << "Got msg from mol: " <<
				molSolveConn_.target( index )->parent()->name() << 
				", " << n << ", " << nInit << 
				", " << mode << ", index = " << index << "\n";
			if ( mode == SOLVER_GET ) {
				molSrc_.sendTo( index, S_[index] );
				cout << " This was a SOLVER_GET operation\n";
			} else if ( mode == SOLVER_SET ) {
				cout << " This was a SOLVER_SET operation to " <<
				molSolveConn_.target( index )->parent()->name() << "\n";
				S_[index] = n;
				Sinit_[index] = nInit;
			} else if ( mode == SOLVER_REBUILD ) {
				rebuildFlag_ = 1;
			}
}
void KsolveWrapper::bufMolFuncLocal( double n, double nInit, int mode, long index )
{
			cout << "Got msg from buffered mol: " <<
				molSolveConn_.target( index )->parent()->name() << 
				", " << n << ", " << nInit << 
				", " << mode << ", index = " << index << "\n";
			if ( mode == SOLVER_GET ) {
				cout << " This was a buffer SOLVER_GET operation, which does nothing.\n";
			} else if ( mode == SOLVER_SET ) {
				cout << " This was a SOLVER_SET operation to buffer " <<
				molSolveConn_.target( index )->parent()->name() << "\n";
				S_[ index + bufOffset_ ] = 
					Sinit_[ index + bufOffset_ ] = nInit;
			}
}
void KsolveWrapper::sumTotMolFuncLocal( double n, double nInit, int mode, long index )
{
			cout << "Got msg from sumtotalled mol: " <<
				molSolveConn_.target( index )->parent()->name() << 
				", " << n << ", " << nInit << 
				", " << mode << ", index = " << index << "\n";
			if ( mode == SOLVER_GET ) {
				sumTotSrc_.sendTo( index, S_[index + sumTotOffset_ ] );
				cout << " This was a sumtotal SOLVER_GET operation\n";
			} else if ( mode == SOLVER_SET ) {
				cout << " This was a SOLVER_SET operation to a sumtotal, which does nothing " <<
				molSolveConn_.target( index )->parent()->name() << "\n";
			}
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processInConnKsolveLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KsolveWrapper, processInConn_ );
	return reinterpret_cast< KsolveWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reinitInConnKsolveLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( KsolveWrapper, reinitInConn_ );
	return reinterpret_cast< KsolveWrapper* >( ( unsigned long )c - OFFSET );
}

