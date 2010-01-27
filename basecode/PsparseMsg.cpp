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

PsparseMsg::PsparseMsg( Element* e1, Element* e2 )
	: SparseMsg( e1, e2 )
{
	;
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

		// This is the crucial line where we define which subset of data
		// can be accessed by this thread.
		row = row * p->numThreadsInGroup + p->threadIndexInGroup;

		const unsigned int* fieldIndex;
		const unsigned int* colIndex;
		unsigned int n = matrix_.getRow( row, &fieldIndex, &colIndex );
		for ( unsigned int j = 0; j < n; ++j ) {
			// Eref tgt( target, DataId( *colIndex++, *fieldIndex++ )
			Eref tgt( e2_, DataId( colIndex[j], fieldIndex[j] ) );
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
		e1->addMsgToConn( m->mid(), srcFinfo->getConnId() );
		e1->addTargetFunc( funcId, srcFinfo->getFuncIndex() );
		m->randomConnect( probability );
		m->loadBalance( numThreadsInGroup );
		return 1;
	}
	return 0;
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
	SparseMatrix< unsigned int > temp = matrix_;
	unsigned int nrows = matrix_.nRows();
	unsigned int ncols = matrix_.nColumns();

	numThreads_ = numThreads;
	matrix_.setSize( numThreads * nrows, ncols ); // Clear and reallocate

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
			matrix_.addRow( i * numThreads + j,
				splitEntry[ j ], splitColIndex[ j ] );
		}
	}
}

// Need a restore function to convert the load-balanced form back into
// the regular sparse matrix.
void PsparseMsg::loadUnbalance()
{
	;
}
