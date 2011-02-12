/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgManager.h"

///////////////////////////////////////////////////////////////////////////

// Static field declaration.
vector< Msg* > Msg::msg_;
vector< MsgId > Msg::garbageMsg_;
vector< unsigned int > Msg::lookupDataId_;
const MsgId Msg::badMsg = 0;
const MsgId Msg::setMsg = 1;

Msg::Msg( MsgId mid, Element* e1, Element* e2, Id managerId )
	: mid_( mid), e1_( e1 ), e2_( e2 )
{
	assert( mid_ < msg_.size() );
	if ( mid_ >= msg_.size() ) {
		msg_.resize( mid + 1, 0 );
		lookupDataId_.resize( mid + 1, 0 );
	}
	msg_[mid_] = this;
	e1->addMsg( mid_ );
	e2->addMsg( mid_ );
	MsgManager::addMsg( mid_, managerId );
}

/**
 * This sets up the set/get msg. It should be called first of all,
 * at init time.
 * Deprecated.
Msg::Msg( Element* e1, Element* e2, MsgId mid, Id managerId )
	: e1_( e1 ), e2_( e2 ), mid_( mid )
{
	if ( msg_.size() < mid )
		msg_.resize( mid + 1 );
	assert( msg_[mid] == 0 );
	assert( lookupDataId_[mid] == 0 );
	msg_[mid] = this;
	lookupDataId_[ mid_ ] = 0;
	e1->addMsg( mid );
	e2->addMsg( mid );
	MsgManager::addMsg( mid_, managerId );
}
 */

Msg::~Msg()
{
	msg_[ mid_ ] = 0;
	e1_->dropMsg( mid_ );
	e2_->dropMsg( mid_ );

	if ( mid_ > 1 )
		garbageMsg_.push_back( mid_ );
}

/**
 * Returns a MsgId for assigning to a new Msg.
 */
MsgId Msg::nextMsgId()
{
	MsgId ret;
	if ( garbageMsg_.size() > 0 ) {
		ret = garbageMsg_.back();
		garbageMsg_.pop_back();
		lookupDataId_[ ret ] = 0;
	} else {
		ret = msg_.size();
		msg_.push_back( 0 );
		lookupDataId_.push_back( 0 );
	}
	return ret;
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
	assert( lookupDataId_.size() == 0 );
	nextMsgId(); // Set aside entry 0 for badMsg;
	nextMsgId(); // Set aside entry 1 for setMsg;
}

void Msg::process( const ProcInfo* p, FuncId fid ) const 
{
	e2_->process( p, fid );
}

const Msg* Msg::getMsg( MsgId m )
{
	assert( m < msg_.size() );
	return msg_[ m ];
}

Msg* Msg::safeGetMsg( MsgId m )
{
	if ( m == badMsg )
		return 0;
	if ( m < msg_.size() )
		return msg_[ m ];
	return 0;
}

Eref Msg::manager( Id id ) const
{
	assert( lookupDataId_.size() > mid_ );
	return Eref( id(), lookupDataId_[ mid_ ] );
}

void Msg::setDataId( unsigned int di ) const
{
	assert( lookupDataId_.size() > mid_ );
	lookupDataId_[ mid_ ] = di;
}

void Msg::addToQ( const Element* src, Qinfo& q,
	const ProcInfo* p, MsgFuncBinding i, const char* arg ) const
{
	if ( e1_ == src ) {
		q.addToQforward( p, i, arg );
	} else {
		q.addToQbackward( p, i, arg ); 
	}
}
