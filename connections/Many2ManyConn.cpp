/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SimpleConn.h"
#include "../utility/SparseMatrix.h"
#include "Many2ManyConn.h"

Many2ManyConnTainer::Many2ManyConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1.e, e2.e, msg1, msg2 ), 
	entries_( e1.e->numEntries(), e2.e->numEntries() ),
	i1_( i1 )
{
	entries_.set( e1.i, e2.i, i2 );
}

Conn* Many2ManyConnTainer::conn( Eref e, unsigned int funcIndex ) const
{
	//	numIter_++; // For reference counting. Do we need it?
	if ( e.e == e1() )
		return new Many2ManyConn( funcIndex, this, e.i );
	else
		return new ReverseMany2ManyConn( funcIndex, this, e.i );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
 
 //TODO array copy
ConnTainer* Many2ManyConnTainer::copy( Element* e1, Element* e2, bool isArray ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );

	Many2ManyConnTainer* ret = 
		new Many2ManyConnTainer( e1, e2, msg1(), msg2(), i1_, 0 );
	ret->entries_ = entries_;
	return ret;
}

/**
 * Add new pair of src and dest element indices to current ConnTainer.
 * Return true on success. Fails if the msg already exists, or
 * if the indices are out of range.
 */
bool Many2ManyConnTainer::addToConnTainer( 
	unsigned int srcEi, unsigned int destEi, unsigned int i2 )
{
	if ( srcEi < entries_.nRows() && destEi < entries_.nColumns() ) {
		entries_.set( srcEi, destEi, i2 );
		return 1;
	}
	return 0;
}

unsigned int Many2ManyConnTainer::getRow( unsigned int i, 
			const unsigned int** index, const unsigned int** eIndex ) const
{
	return entries_.getRow( i, index, eIndex );
}

unsigned int Many2ManyConnTainer::getColumn( unsigned int i, 
			vector< unsigned int >& index,
			vector< unsigned int >& eIndex ) const
{
	return entries_.getColumn( i, index, eIndex );
}

// # of srcs that terminate on the destEindex.
unsigned int Many2ManyConnTainer::numSrc( unsigned int destEindex ) const
{
	vector< unsigned int > index;
	vector< unsigned int > eIndex;
	return entries_.getColumn( destEindex, index, eIndex );
}

// # of targets starting at srcEindex
unsigned int Many2ManyConnTainer::numDest( unsigned int srcEindex ) const
{
	const unsigned int* index;
	const unsigned int* eIndex;
	return entries_.getRow( srcEindex, &index, &eIndex );
}

//////////////////////////////////////////////////////////////////////
//  Many2ManyConn
//////////////////////////////////////////////////////////////////////

const Conn* Many2ManyConn::flip( unsigned int funcIndex ) const
{
	return new ReverseMany2ManyConn( funcIndex, s_, *tgtEindexIter_ );
}
