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
#include "../randnum/randnum.h"
#include "../biophysics/Synapse.h"

//////////////////////////////////////////////////////////////////
//    MOOSE wrapper functions for field access.
//////////////////////////////////////////////////////////////////

const Cinfo* SparseMsgWrapper::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< SparseMsgWrapper, unsigned int > numRows(
		"numRows",
		"Number of rows in matrix.",
		&SparseMsgWrapper::getNumRows
	);
	static ReadOnlyValueFinfo< SparseMsgWrapper, unsigned int > numColumns(
		"numColumns",
		"Number of columns in matrix.",
		&SparseMsgWrapper::getNumColumns
	);
	static ReadOnlyValueFinfo< SparseMsgWrapper, unsigned int > numEntries(
		"numEntries",
		"Number of Entries in matrix.",
		&SparseMsgWrapper::getNumEntries
	);

	static ValueFinfo< SparseMsgWrapper, double > probability(
		"probability",
		"connection probability for random connectivity.",
		&SparseMsgWrapper::setProbability,
		&SparseMsgWrapper::getProbability
	);

	static ValueFinfo< SparseMsgWrapper, long > seed(
		"seed",
		"Random number seed for generating probabilistic connectivity.",
		&SparseMsgWrapper::setSeed,
		&SparseMsgWrapper::getSeed
	);

////////////////////////////////////////////////////////////////////////
// DestFinfos
////////////////////////////////////////////////////////////////////////

	static DestFinfo setRandomConnectivity( "setRandomConnectivity",
		"Assigns connectivity with specified probability and seed",
		new OpFunc2< SparseMsgWrapper, double, long >( 
		&SparseMsgWrapper::setRandomConnectivity ) );

	static DestFinfo setEntry( "setEntry",
		"Assigns single row,column value",
		new OpFunc3< SparseMsgWrapper, unsigned int, unsigned int, unsigned int >( 
		&SparseMsgWrapper::setEntry ) );

	static DestFinfo unsetEntry( "unsetEntry",
		"Clears single row,column entry",
		new OpFunc2< SparseMsgWrapper, unsigned int, unsigned int >( 
		&SparseMsgWrapper::unsetEntry ) );

	static DestFinfo clear( "clear",
		"Clears out the entire matrix",
		new OpFunc0< SparseMsgWrapper >( 
		&SparseMsgWrapper::clear ) );

	static DestFinfo transpose( "transpose",
		"Transposes the sparse matrix",
		new OpFunc0< SparseMsgWrapper >( 
		&SparseMsgWrapper::transpose ) );

	static DestFinfo loadBalance( "loadBalance",
		"Decomposes the sparse matrix for threaded operation",
		new OpFunc1< SparseMsgWrapper, unsigned int >( 
		&SparseMsgWrapper::loadBalance ) );

	static DestFinfo loadUnbalance( "loadUnbalance",
		"Converts the threaded matrix back into single-thread form",
		new OpFunc0< SparseMsgWrapper >( 
		&SparseMsgWrapper::loadUnbalance ) );

////////////////////////////////////////////////////////////////////////
// Assemble it all.
////////////////////////////////////////////////////////////////////////

	static Finfo* sparseMsgFinfos[] = {
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

	static Cinfo sparseMsgCinfo (
		"SparseMsg",					// name
		MsgManager::initCinfo(),		// base class
		sparseMsgFinfos,
		sizeof( sparseMsgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< SparseMsgWrapper >()
	);

	return &sparseMsgCinfo;
}

static const Cinfo* sparseMsgCinfo = SparseMsgWrapper::initCinfo();

//////////////////////////////////////////////////////////////////
//    Value Fields
//////////////////////////////////////////////////////////////////
void SparseMsgWrapper::setProbability ( double probability )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		p_ = probability;
		mtseed( seed_ );
		pm->randomConnect( probability );
	}
}

double SparseMsgWrapper::getProbability ( ) const
{
	return p_;
}

void SparseMsgWrapper::setSeed ( long seed )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		seed_ = seed;
		mtseed( seed_ );
		pm->randomConnect( p_ );
	}
}

long SparseMsgWrapper::getSeed () const
{
	return seed_;
}

unsigned int SparseMsgWrapper::getNumRows() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		return pm->getMatrix().nRows();
	}
	return 0;
}

unsigned int SparseMsgWrapper::getNumColumns() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		return pm->getMatrix().nColumns();
	}
	return 0;
}

unsigned int SparseMsgWrapper::getNumEntries() const
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		return pm->getMatrix().nEntries();
	}
	return 0;
}

//////////////////////////////////////////////////////////////////
//    DestFields
//////////////////////////////////////////////////////////////////

void SparseMsgWrapper::setRandomConnectivity(
	double probability, long seed )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg* >( m );
	if ( pm ) {
		p_ = probability;
		seed_ = seed;
		mtseed( seed );
		pm->randomConnect( probability );
	}
}

void SparseMsgWrapper::setEntry(
	unsigned int row, unsigned int column, unsigned int value )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->getMatrix().set( row, column, value );
	}
}

void SparseMsgWrapper::unsetEntry( unsigned int row, unsigned int column )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->getMatrix().unset( row, column );
	}
}

void SparseMsgWrapper::clear()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->getMatrix().clear();
	}
}

void SparseMsgWrapper::transpose()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->getMatrix().transpose();
	}
}

void SparseMsgWrapper::loadBalance( unsigned int numThreads )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->loadBalance( numThreads );
	}
}

void SparseMsgWrapper::loadUnbalance()
{
	Msg* m = Msg::safeGetMsg( getMid() );
	SparseMsg* pm = dynamic_cast< SparseMsg *>( m );
	if ( pm ) {
		pm->loadUnbalance();
	}
}

//////////////////////////////////////////////////////////////////
//    Here are the actual class functions
//////////////////////////////////////////////////////////////////


SparseMsg::SparseMsg( Element* e1, Element* e2 )
	: Msg( e1, e2, id_ ),
	matrix_( e1->dataHandler()->numData1(), e2->dataHandler()->numData1() )
{
	assert( e1->dataHandler()->numDimensions() == 1  && 
		e2->dataHandler()->numDimensions() >= 1 );
}

SparseMsg::~SparseMsg()
{
	MsgManager::dropMsg( mid() );
}

unsigned int rowIndex( const Element* e, const DataId& d )
{
	if ( e->dataHandler()->numDimensions() == 1 ) {
		return d.data();
	} else if ( e->dataHandler()->numDimensions() == 2 ) {
		// This is a nasty case, hopefully very rare.
		unsigned int row = 0;
		for ( unsigned int i = 0; i < d.data(); ++i )
			row += e->dataHandler()->numData2( i );
		return ( row + d.field() );
	}
	return 0;
}

void SparseMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	// arg += sizeof( Qinfo );

	/**
	 * The system is really optimized for data from e1 to e2.
	 */
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		unsigned int row = rowIndex( e1_, q->srcIndex() );
		// unsigned int oldRow = row;


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
				":Sparse exec    [" << colIndex[j] << 
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
 * Should really have a seed argument
 */
unsigned int SparseMsg::randomConnect( double probability )
{
	unsigned int nRows = matrix_.nRows(); // Sources
	unsigned int nCols = matrix_.nColumns();	// Destinations
	// matrix_.setSize( 0, nRows ); // we will transpose this later.
	matrix_.clear();
	unsigned int totalSynapses = 0;
	unsigned int startSynapse = 0;
	vector< unsigned int > sizes;
	bool isFirstRound = 1;
	unsigned int totSynNum = 0;

	// SynElement* syn = dynamic_cast< SynElement* >( e2_ );
	Element* syn = e2_;
	syn->dataHandler()->getNumData2( sizes );
	assert( sizes.size() == nCols );

	for ( unsigned int i = 0; i < nCols; ++i ) {
		// Check if synapse is on local node
		bool isSynOnMyNode = syn->dataHandler()->isDataHere( i );
		vector< unsigned int > synIndex;
		// This needs to be obtained from current size of syn array.
		// unsigned int synNum = sizes[ i ];
		unsigned int synNum = 0;
		for ( unsigned int j = 0; j < nRows; ++j ) {
			double r = mtrand(); // Want to ensure it is called each time round the loop.
			if ( isSynOnMyNode ) {
				if ( isFirstRound ) {
					startSynapse = totSynNum;
					isFirstRound = 0;
				}
			}
			if ( r < probability && isSynOnMyNode ) {
				synIndex.push_back( synNum );
				++synNum;
			} else {
				synIndex.push_back( ~0 );
			}
			if ( r < probability )
				++totSynNum;
		}
		sizes[ i ] = synNum;
		totalSynapses += synNum;

		matrix_.addRow( i, synIndex );
	}
	syn->dataHandler()->setNumData2( startSynapse, sizes );
	// cout << Shell::myNode() << ": sizes.size() = " << sizes.size() << ", ncols = " << nCols << ", startSynapse = " << startSynapse << endl;
	matrix_.transpose();
	return totalSynapses;
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
void SparseMsg::loadBalance( unsigned int numThreads )
{
	sparseMatrixBalance( numThreads, matrix_ );
	numThreads_ = numThreads;
}

// Need a restore function to convert the load-balanced form back into
// the regular sparse matrix.
void SparseMsg::loadUnbalance()
{
	;
}

Id SparseMsg::id() const
{
	return id_;
}

void SparseMsg::setMatrix( const SparseMatrix< unsigned int >& m )
{
	matrix_ = m;
}

SparseMatrix< unsigned int >& SparseMsg::getMatrix( )
{
	return matrix_;
}
