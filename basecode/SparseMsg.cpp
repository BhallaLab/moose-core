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

SparseMsg::SparseMsg( Element* e1, Element* e2 )
	: Msg( e1, e2 ), matrix_( e1->numData1(), e2->numData1() )
{
	assert( e1->numDimensions() == 1  && e2->numDimensions() >= 1 );
}

void SparseMsg::exec( Element* target, const char* arg ) const
{
	assert( target == e1_ || target == e2_ );

	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	// arg += sizeof( Qinfo );
	const OpFunc* f = target->cinfo()->getOpFunc( q->fid() );

	/**
	 * The system is really optimized for data from e1 to e2.
	 */
	if ( target == e2_ ) {
		unsigned int row = q->srcIndex();
		const unsigned int* fieldIndex;
		const unsigned int* colIndex;
		unsigned int n = matrix_.getRow( row, &fieldIndex, &colIndex );
		for ( unsigned int j = 0; j < n; ++j ) {
			// Eref tgt( target, DataId( *colIndex++, *fieldIndex++ )
			Eref tgt( target, DataId( colIndex[j], fieldIndex[j] ) );
			f->op( tgt, arg );
		}
	} else { // Avoid using this back operation!
		// Note that we do NOT use the fieldIndex going backward. It is
		// assumed that e1 does not have fieldArrays.
		unsigned int column = q->srcIndex();
		vector< unsigned int > fieldIndex;
		vector< unsigned int > rowIndex;
		unsigned int n = matrix_.getColumn( column, fieldIndex, rowIndex );
		for ( unsigned int j = 0; j < n; ++j ) {
			Eref tgt( target, DataId( rowIndex[j] ) );
			f->op( tgt, arg );
		}
	}
}

bool SparseMsg::add( Element* e1, const string& srcField, 
			Element* e2, const string& destField, double probability )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1, srcField,
		e2, destField, funcId );

	if ( srcFinfo ) {
		SparseMsg* m = new SparseMsg( e1, e2 );
		e1->addMsgToConn( m, srcFinfo->getConnId() );
		e1->addTargetFunc( funcId, srcFinfo->getFuncIndex() );
		m->randomConnect( probability );
		return 1;
	}
	return 0;
}

unsigned int SparseMsg::randomConnect( double probability )
{
	unsigned int nRows = matrix_.nRows();
	unsigned int nCols = matrix_.nColumns();
	// matrix_.setSize( 0, nRows ); // we will transpose this later.
	matrix_.clear();
	unsigned int totalSynapses = 0;
	vector< unsigned int > sizes;

	SynElement* syn = dynamic_cast< SynElement* >( e2_ );
	assert( syn );
	syn->getArraySizes( sizes );
	assert( sizes.size() == nCols );

	for ( unsigned int i = 0; i < nCols; ++i ) {
		vector< unsigned int > synIndex;
		// This needs to be obtained from current size of syn array.
		unsigned int synNum = sizes[ i ];
		for ( unsigned int j = 0; j < nRows; ++j ) {
			if ( mtrand() < probability ) {
				synIndex.push_back( synNum );
				++synNum;
			} else {
				synIndex.push_back( ~0 );
			}
		}
		sizes[ i ] = synNum;
		/**
		 * Here I have a problem. The number of synapses is known here, as
		 * synIndex. I need to specify to the target base Element to 
		 * assign this number, without knowing what type this base Element
		 * is.
		 */
		totalSynapses += synNum;

		matrix_.addRow( i, synIndex );
	}
	syn->setArraySizes( sizes );
	matrix_.transpose();
	return totalSynapses;
}
