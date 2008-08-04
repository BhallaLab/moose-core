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
#include "TraverseMsgConn.h"


//////////////////////////////////////////////////////////////////////
//  TraverseMsgConn
//////////////////////////////////////////////////////////////////////

TraverseMsgConn::TraverseMsgConn( const Msg* m, const Element* e,
	unsigned int eIndex )
	: msg_( m ), e_( e ), eIndex_( eIndex ), c_( 0 ), cti_( (vector <ConnTainer*>::const_iterator )0 )
{
	// Advance till we get to a 'good' conn, or to the end.
	for ( mi_ = msg_; mi_; mi_ = mi_->next( e ) ) {
		for ( cti_ = mi_->begin(); cti_ != mi_->end(); cti_++ ) {
			c_ = ( *cti_ )->conn( eIndex, mi_->isDest() ); 
			for ( ; c_->good(); c_->increment() )
					return;
		}
	}
	// A non-starter.
	c_ = 0;
}

TraverseMsgConn::~TraverseMsgConn()
{
	if ( c_ != 0 )
		delete c_;
}

void TraverseMsgConn::increment()
{
	if ( c_ == 0 ) // already at end.
		return;
	assert( c_->good() );
	assert( mi_ != 0 );
	assert( cti_ != mi_->end() );

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
	for ( ; cti_ != mi_->end(); cti_++ ) {
		c_ = ( *cti_ )->conn( eIndex_, mi_->isDest() ); 
		for ( ; c_->good(); c_->increment() )
				return;
	}

	/////////////////////////////////////////////////
	// Need to advance to next Msg
	/////////////////////////////////////////////////
	if ( c_ )
		delete c_;
	c_ = 0;
	for ( mi_ = mi_->next( e_ ); mi_; mi_ = mi_->next( e_ ) ) {
		for ( cti_ = mi_->begin(); cti_ != mi_->end(); cti_++ ) {
			c_ = ( *cti_ )->conn( eIndex_, mi_->isDest() ); 
			for ( ; c_->good(); c_->increment() )
					return;
		}
	}

	/////////////////////////////////////////////////
	// All hope is lost. Bail out.
	/////////////////////////////////////////////////

	if ( c_ )
		delete c_;
	c_ = 0;
}

void TraverseMsgConn::nextElement()
{
	if ( c_ == 0 ) // already at end.
		return;
	assert( c_->good() );
	assert( mi_ != 0 );
	assert( cti_ != mi_->end() );

	/////////////////////////////////////////////////
	// Need to advance to next ConnTainer
	/////////////////////////////////////////////////
	delete c_;
	c_ = 0;
	cti_++;
	for ( ; cti_ != mi_->end(); cti_++ ) {
		c_ = ( *cti_ )->conn( eIndex_, mi_->isDest() ); 
		for ( ; c_->good(); c_->increment() )
				return;
	}

	/////////////////////////////////////////////////
	// Need to advance to next Msg
	/////////////////////////////////////////////////
	if ( c_ )
		delete c_;
	c_ = 0;
	for ( mi_ = mi_->next( e_ ); mi_; mi_ = mi_->next( e_ ) ) {
		for ( cti_ = mi_->begin(); cti_ != mi_->end(); cti_++ ) {
			c_ = ( *cti_ )->conn( eIndex_, mi_->isDest() ); 
			for ( ; c_->good(); c_->increment() )
					return;
		}
	}

	/////////////////////////////////////////////////
	// All hope is lost. Bail out.
	/////////////////////////////////////////////////

	if ( c_ )
		delete c_;
	c_ = 0;

}


bool TraverseMsgConn::good() const
{
	if ( c_ == 0 )
		return 0;
	return c_->good();
}

const Conn* TraverseMsgConn::flip() const
{
	return new TraverseMsgConn( msg_, e_, eIndex_ );
}

const ConnTainer* TraverseMsgConn::connTainer() const
{
	if ( good() )
		return *cti_;
	return 0;
}
