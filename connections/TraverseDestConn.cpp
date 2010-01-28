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
#include "TraverseDestConn.h"


//////////////////////////////////////////////////////////////////////
//  TraverseDestConn
//////////////////////////////////////////////////////////////////////

TraverseDestConn::TraverseDestConn(
	const vector< ConnTainer* >* ct, Eref e )
	: Conn( 0 ), ct_( ct ), c_( 0 ), e_( e )
{
	assert( ct_ != 0 );
	// Advance till we get to a 'good' conn, or to the end.
	for ( cti_ = ct_->begin(); cti_ != ct_->end(); cti_++ ) {
		c_ = ( *cti_ )->conn( e, 0 ); 
		for ( ; c_->good(); c_->increment() )
			return;
	}
	// A non-starter.
	c_ = 0;
}

TraverseDestConn::~TraverseDestConn()
{
	if ( c_ != 0 )
		delete c_;
}


void TraverseDestConn::increment()
{
	if ( c_ == 0 ) // already at end.
		return;
	assert( c_->good() );
	assert( ct_ != 0 );
	assert( cti_ != ct_->end() );

	/////////////////////////////////////////////////
	// Need to advance to next Conn
	/////////////////////////////////////////////////
	c_->increment();
	if ( c_->good() ) return;

	/////////////////////////////////////////////////
	// Need to advance to next ConnTainer
	/////////////////////////////////////////////////
	delete c_;
	c_ = 0;
	cti_++;
	for ( ; cti_ != ct_->end(); cti_++ ) {
		c_ = ( *cti_ )->conn( e_, 0 ); 
		for ( ; c_->good(); c_->increment() )
				return;
	}

	/////////////////////////////////////////////////
	// All hope is lost. Bail out.
	/////////////////////////////////////////////////

	if ( c_ )
		delete c_;
	c_ = 0;
}

void TraverseDestConn::nextElement()
{
	if ( c_ == 0 ) // already at end.
		return;
	assert( c_->good() );
	assert( ct_ != 0 );
	assert( cti_ != ct_->end() );

	/////////////////////////////////////////////////
	// Need to advance to next ConnTainer
	/////////////////////////////////////////////////
	delete c_;
	c_ = 0;
	cti_++;
	for ( ; cti_ != ct_->end(); cti_++ ) {
		c_ = ( *cti_ )->conn( e_, 0 ); 
		for ( ; c_->good(); c_->increment() )
				return;
	}

	/////////////////////////////////////////////////
	// All hope is lost. Bail out.
	/////////////////////////////////////////////////

	if ( c_ )
		delete c_;
	c_ = 0;

}

bool TraverseDestConn::good() const
{
	if ( c_ == 0 )
		return 0;
	return c_->good();
}

// Doesn't really do anything.
const Conn* TraverseDestConn::flip( unsigned int funcIndex ) const
{
	return new TraverseDestConn( ct_, e_ );
}

const ConnTainer* TraverseDestConn::connTainer() const
{
	if ( good() )
		return *cti_;
	return 0;
}
