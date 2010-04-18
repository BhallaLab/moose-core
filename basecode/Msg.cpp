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

///////////////////////////////////////////////////////////////////////////

// Static field declaration.
vector< Msg* > Msg::msg_;
vector< MsgId > Msg::garbageMsg_;
const MsgId Msg::badMsg = 0;
const MsgId Msg::setMsg = 1;

Msg::Msg( Element* e1, Element* e2 )
	: e1_( e1 ), e2_( e2 )
{
	if ( garbageMsg_.size() > 0 ) {
		mid_ = garbageMsg_.back();
		garbageMsg_.pop_back();
		msg_[mid_] = this;
	} else {
		mid_ = msg_.size();
		msg_.push_back( this );
	}
	e1->addMsg( mid_ );
	e2->addMsg( mid_ );
}

/**
 * This sets up the set/get msg. It should be called first of all,
 * at init time.
 */
Msg::Msg( Element* e1, Element* e2, MsgId mid )
	: e1_( e1 ), e2_( e2 )
{
	if ( msg_.size() < mid )
		msg_.resize( mid + 1 );
	msg_[mid_] = this;
	e1->addMsg( mid_ );
	e2->addMsg( mid_ );
}

Msg::~Msg()
{
	msg_[ mid_ ] = 0;
	e1_->dropMsg( mid_ );
	e2_->dropMsg( mid_ );

	garbageMsg_.push_back( mid_ );
}

void Msg::deleteMsg( MsgId mid )
{
	assert( mid < msg_.size() );
	Msg* m = msg_[ mid ];
	if ( m != 0 )
		delete m;
	msg_[ mid ] = 0;
}

/**
 * Initialize the Null location in the Msg vector.
 */
void Msg::initNull()
{
	assert( msg_.size() == 0 );
	msg_.push_back( 0 );
}

/*
void Msg::clearQ() const 
{
	e2_->clearQ();
}
*/

void Msg::process( const ProcInfo* p ) const 
{
	e2_->process( p );
}

const Msg* Msg::getMsg( MsgId m )
{
	assert( m < msg_.size() );
	return msg_[ m ];
}
