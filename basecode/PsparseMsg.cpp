/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Message.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "PsparseMsg.h"
#include "../randnum/randnum.h"
#include "../biophysics/Synapse.h"

//////////////////////////////////////////////////////////////////
//    MOOSE wrapper functions for field access.
//////////////////////////////////////////////////////////////////

const Cinfo* PsparseMsgWrapper::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	/*
	static ReadOnlyValueFinfo< PsparseMsgWrapper, Id > element1(
		"e1",
		"Id of source Element.",
		&PsparseMsgWrapper::getE1
	);
	static ReadOnlyValueFinfo< PsparseMsgWrapper, Id > element2(
		"e2",
		"Id of source Element.",
		&PsparseMsgWrapper::getE2
	);
	*/
	static ReadOnlyValueFinfo< PsparseMsgWrapper, unsigned int > numRows(
		"numRows",
		"Number of rows in matrix.",
		&PsparseMsgWrapper::getNumRows
	);
	static ReadOnlyValueFinfo< PsparseMsgWrapper, unsigned int > numColumns(
		"numColumns",
		"Number of columns in matrix.",
		&PsparseMsgWrapper::getNumColumns
	);
	static ReadOnlyValueFinfo< PsparseMsgWrapper, unsigned int > numEntries(
		"numEntries",
		"Number of Entries in matrix.",
		&PsparseMsgWrapper::getNumEntries
	);

	static ValueFinfo< PsparseMsgWrapper, double > probability(
		"probability",
		"connection probability for random connectivity.",
		&PsparseMsgWrapper::setProbability,
		&PsparseMsgWrapper::getProbability
	);

	static ValueFinfo< PsparseMsgWrapper, long > seed(
		"seed",
		"Random number seed for generating probabilistic connectivity.",
		&PsparseMsgWrapper::setSeed,
		&PsparseMsgWrapper::getSeed
	);

////////////////////////////////////////////////////////////////////////
// DestFinfos
////////////////////////////////////////////////////////////////////////

	static DestFinfo setRandomConnectivity( "setRandomConnectivity",
		"Assigns connectivity with specified probability and seed",
		new OpFunc2< PsparseMsgWrapper, double, long >( 
		&PsparseMsgWrapper::setRandomConnectivity ) );

	static DestFinfo setEntry( "setEntry",
		"Assigns single row,column value",
		new OpFunc3< PsparseMsgWrapper, unsigned int, unsigned int, unsigned int >( 
		&PsparseMsgWrapper::setEntry ) );

	static DestFinfo unsetEntry( "unsetEntry",
		"Clears single row,column entry",
		new OpFunc2< PsparseMsgWrapper, unsigned int, unsigned int >( 
		&PsparseMsgWrapper::unsetEntry ) );

	static DestFinfo clear( "clear",
		"Clears out the entire matrix",
		new OpFunc0< PsparseMsgWrapper >( 
		&PsparseMsgWrapper::clear ) );

	static DestFinfo transpose( "transpose",
		"Transposes the sparse matrix",
		new OpFunc0< PsparseMsgWrapper >( 
		&PsparseMsgWrapper::transpose ) );

	static DestFinfo loadBalance( "loadBalance",
		"Decomposes the sparse matrix for threaded operation",
		new OpFunc1< PsparseMsgWrapper, unsigned int >( 
		&PsparseMsgWrapper::loadBalance ) );

	static DestFinfo loadUnbalance( "loadUnbalance",
		"Converts the threaded matrix back into single-thread form",
		new OpFunc0< PsparseMsgWrapper >( 
		&PsparseMsgWrapper::loadUnbalance ) );

////////////////////////////////////////////////////////////////////////
// Assemble it all.
////////////////////////////////////////////////////////////////////////

	static Finfo* pSparseMsgFinfos[] = {
		&numRows,			// readonly value
		&numColumns,		// readonly value
		&numEntries,		// readonly value
		&probability,		// value
		&seed,				// value
		&setRandomConnectivity,	// dest
		&setEntry,			// dest
		&unsetEntry,		//dest
		&clear,				//dest
		&transpose,			//dest
		&loadBalance,		//dest
		&loadUnbalance		//dest
	};

	static Cinfo pSparseMsgCinfo (
		"PsparseMsg",					// name
		MsgManager::initCinfo(),		// base class
		pSparseMsgFinfos,
		sizeof( pSparseMsgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< PsparseMsgWrapper >()
	);

	return &pSparseMsgCinfo;
}

static const Cinfo* pSparseMsgCinfo = PsparseMsgWrapper::initCinfo();

//////////////////////////////////////////////////////////////////
//    Value Fields
//////////////////////////////////////////////////////////////////
void PsparseMsgWrapper::setProbability ( double probability )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		p_ = probability;
		mtseed( seed_ );
		pm->randomConnect( probability );
	}
}

double PsparseMsgWrapper::getProbability ( ) const
{
	return p_;
}

void PsparseMsgWrapper::setSeed ( long seed )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		seed_ = seed;
		mtseed( seed_ );
		pm->randomConnect( p_ );
	}
}

long PsparseMsgWrapper::getSeed () const
{
	return seed_;
}

unsigned int PsparseMsgWrapper::getNumRows() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		return pm->matrix().nRows();
	}
	return 0;
}

unsigned int PsparseMsgWrapper::getNumColumns() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		return pm->matrix().nColumns();
	}
	return 0;
}

unsigned int PsparseMsgWrapper::getNumEntries() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		return pm->matrix().nEntries();
	}
	return 0;
}

//////////////////////////////////////////////////////////////////
//    DestFields
//////////////////////////////////////////////////////////////////

void PsparseMsgWrapper::setRandomConnectivity(
	double probability, long seed )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg* >( m );
	if ( pm ) {
		p_ = probability;
		seed_ = seed;
		mtseed( seed );
		pm->randomConnect( probability );
	}
}

void PsparseMsgWrapper::setEntry(
	unsigned int row, unsigned int column, unsigned int value )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->matrix().set( row, column, value );
	}
}

void PsparseMsgWrapper::unsetEntry( unsigned int row, unsigned int column )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->matrix().unset( row, column );
	}
}

void PsparseMsgWrapper::clear()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->matrix().clear();
	}
}

void PsparseMsgWrapper::transpose()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->matrix().transpose();
	}
}

void PsparseMsgWrapper::loadBalance( unsigned int numThreads )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->loadBalance( numThreads );
	}
}

void PsparseMsgWrapper::loadUnbalance()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	PsparseMsg* pm = dynamic_cast< PsparseMsg *>( m );
	if ( pm ) {
		pm->loadUnbalance();
	}
}

//////////////////////////////////////////////////////////////////
//    Here are the actual class functions
//////////////////////////////////////////////////////////////////


PsparseMsg::PsparseMsg( Element* e1, Element* e2 )
	: SparseMsg( e1, e2, id_ )
{
	;
}

PsparseMsg::~PsparseMsg()
{
	MsgManager::dropMsg( mid() );
}

void PsparseMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	// arg += sizeof( Qinfo );

	/**
	 * The system is really optimized for data from e1 to e2.
	 */
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		unsigned int row = rowIndex( e1_, q->srcIndex() );
		unsigned int oldRow = row;


		// This is the crucial line where we define which subset of data
		// can be accessed by this thread.
		row = row * p->numThreadsInGroup + p->threadIndexInGroup;

		const unsigned int* fieldIndex;
		const unsigned int* colIndex;
		unsigned int n = matrix_.getRow( row, &fieldIndex, &colIndex );

		/*
		if ( oldRow % 100 == 0 ) {
			cout << Shell::myNode() << "." << p->threadIndexInGroup << 
				": row = " << oldRow << 
				", Trow = " << row <<
				", n = " << n << 
				", t = " << p->currTime <<
				endl;
		}
		for ( unsigned int j = 0; j < n; ++j ) {
			cout << Shell::myNode() << "." << p->threadIndexInGroup << 
			": " << oldRow << 
			" colindex[" << j << "] = " << colIndex[j] <<
			", fieldindex[" << j << "] = " << fieldIndex[j] << 
			endl << flush;
		}
		*/

		// J counts over all the column entries, i.e., all targets.
		for ( unsigned int j = 0; j < n; ++j ) {
			Eref tgt( e2_, DataId( colIndex[j], fieldIndex[j] ) );
			/*
			if ( colIndex[j] % 100 == 0 ) {
				cout << Shell::myNode() << "." << p->threadIndexInGroup << 
				":Psparse exec    [" << colIndex[j] << 
				"," << fieldIndex[j] << 
				"], target here = " << tgt.isDataHere() <<
				", t = " << p->currTime << endl << flush;
			}
			*/
			if ( tgt.isDataHere() )
				f->op( tgt, arg );
		}
	} else  if ( p->threadIndexInGroup == 0 ) {
		// Avoid using this back operation! Currently we don't
		// even try to do it backward with threading.
		// Note that we do NOT use the fieldIndex going backward. It is
		// assumed that e1 does not have fieldArrays.
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		unsigned int column = rowIndex( e2_, q->srcIndex() );
		vector< unsigned int > fieldIndex;
		vector< unsigned int > rowIndex;
		unsigned int n = matrix_.getColumn( column, fieldIndex, rowIndex );
		for ( unsigned int j = 0; j < n; ++j ) {
			Eref tgt( e1_, DataId( rowIndex[j] ) );
			if ( tgt.isDataHere() )
				f->op( tgt, arg );
		}
	}
}

/**
 * This mostly duplicates what the SparseMsg variant does, but since
 * it explicitly creates a SparseMsg we can't just use the parent func.
 * Then it does the load balancing.
 */
bool PsparseMsg::add( Element* e1, const string& srcField, 
	Element* e2, const string& destField, double probability, 
	unsigned int numThreadsInGroup )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1, srcField,
		e2, destField, funcId );

	if ( srcFinfo ) {
		PsparseMsg* m = new PsparseMsg( e1, e2 );
		e1->addMsgAndFunc( m->mid(), funcId, srcFinfo->getBindIndex() );
		m->randomConnect( probability );
		m->loadBalance( numThreadsInGroup );
		return 1;
	}
	return 0;
}

// Utility function for doing load balance
void sparseMatrixBalance( 
	unsigned int numThreads, SparseMatrix< unsigned int >& matrix )
{
	if ( numThreads <= 1 )
		return;
	SparseMatrix< unsigned int > temp = matrix;
	unsigned int nrows = matrix.nRows();
	unsigned int ncols = matrix.nColumns();

	matrix.setSize( numThreads * nrows, ncols ); // Clear and reallocate

	for ( unsigned int i = 0; i < temp.nRows(); ++i )
	{
		const unsigned int* entry; // I thought it was DataIds.
		const unsigned int* colIndex;
		unsigned int numEntries = temp.getRow( i, &entry, &colIndex );
		vector< vector< unsigned int > > splitEntry( numThreads );
		vector< vector< unsigned int > > splitColIndex( numThreads );
		for ( unsigned int j = 0; j < numEntries; ++j ) {
			unsigned int targetThread = ( colIndex[j] * numThreads) / ncols;
			assert( targetThread < numThreads );
			splitEntry[ targetThread ].push_back( entry[ j ] );
			splitColIndex[ targetThread ].push_back( colIndex[ j ] );
		}
		for ( unsigned int j = 0; j < numThreads; ++j ) {
			matrix.addRow( i * numThreads + j,
				splitEntry[ j ], splitColIndex[ j ] );
		}
	}
}

/**
 * loadBalance: 
 * Splits up the sparse matrix so that any given colIndex will occur
 * only on one subset. This ensures that only a single thread will 
 * ever write to a give target, specified by that colIndex.
 *
 * The subsets are accessed sequentially: For source Element 0 we do 
 * thread0, thread1, thread2...
 * then again for source Element 1 we do thread0, thread1, thread2 ...
 * and so on.
 *
 */
void PsparseMsg::loadBalance( unsigned int numThreads )
{
	sparseMatrixBalance( numThreads, matrix_ );
	numThreads_ = numThreads;
}

// Need a restore function to convert the load-balanced form back into
// the regular sparse matrix.
void PsparseMsg::loadUnbalance()
{
	;
}

Id PsparseMsg::id() const
{
	return id_;
}
