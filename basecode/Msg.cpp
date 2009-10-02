/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Qinfo.h"

///////////////////////////////////////////////////////////////////////////
Msg::Msg( Element* e1, Element* e2 )
	: e1_( e1 ), e2_( e2 )
{
	m1_ = e1->addMsg( this );
	m2_ = e2->addMsg( this );
}

Msg::~Msg()
{
	;
}

void Msg::clearQ() const 
{
	e1_->clearQ();
}

void Msg::process( const ProcInfo* p ) const 
{
	e1_->process( p );
}

///////////////////////////////////////////////////////////////////////////

/*
SparseMsg::SparseMsg( Element* src, Element* dest )
	: Msg( src, dest )
{
	;
}

void SparseMsg::addSpike( unsigned int srcElementIndex, double time ) const
{
	const unsigned int* synIndex;
	const unsigned int* elementIndex;
	unsigned int n = m_.getRow( srcElementIndex, &synIndex, &elementIndex );
	for ( unsigned int i = 0; i < n; ++i )
		dest_->addSpike( *elementIndex++, *synIndex++, time );
}
*/

///////////////////////////////////////////////////////////////////////////

/*
One2OneMsg::One2OneMsg( Element* src, Element* dest )
	: Msg( src, dest ), synIndex_( 0 )
{
	;
}

void One2OneMsg::addSpike( unsigned int srcElementIndex, double time ) const
{
	dest_->addSpike( srcElementIndex, synIndex_, time );
}



*/
///////////////////////////////////////////////////////////////////////////

SingleMsg::SingleMsg( Eref e1, Eref e2 )
	: Msg( e1.element(), e2.element() ),
	i1_( e1.index() ), 
	i2_( e2.index() )
{
	;
}

void SingleMsg::addToQ( const Element* caller, FuncId f, 
			const char* arg, unsigned int size ) const
{
	if ( caller == e1_ ) {
		e2_->addToQ( Qinfo( f, size, m2_ ), arg );
	} else {
		assert( caller == e2_ );
		e1_->addToQ( Qinfo( f, size, m1_ ), arg );
	}
}

const char* SingleMsg::exec( Element* target, OpFunc* f, 
			const char* arg ) const
{
	if ( target == e1_ ) {
		f->op( Eref( target, i1_ ), arg );
	} else {
		assert( target == e2_ );
		f->op( Eref( target, i2_ ), arg );
	}
	return 0;
}
