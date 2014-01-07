/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OneToOneMsg.h"

// Initializing static variables
Id OneToOneMsg::managerId_;
vector< OneToOneMsg* > OneToOneMsg::msg_;

OneToOneMsg::OneToOneMsg( Element* e1, Element* e2, unsigned int msgIndex )
	: Msg( ObjId( managerId_, (msgIndex != 0) ? msgIndex: msg_.size() ),
					e1, e2 )
{
	if ( msgIndex == 0 ) {
		msg_.push_back( this );
	} else {
		if ( msg_.size() <= msgIndex )
			msg_.resize( msgIndex + 1 );
		msg_[ msgIndex ] = this;
	}
}

OneToOneMsg::~OneToOneMsg()
{
	assert( mid_.dataIndex < msg_.size() );
	msg_[ mid_.dataIndex ] = 0; // ensure deleted ptr isn't reused.
}

/**
 * This is a little tricky because we might be mapping between
 * data entries and field entries here.
 * May wish also to apply to exec operations.
 * At this point, the effect of trying to go between regular
 * data entries and field entries is undefined.
 */
Eref OneToOneMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ ) {
		return Eref( e2_, src.dataIndex() );
	} else if ( src.element() == e2_ ) {
		return Eref( e1_, src.dataIndex() );
	}
	return Eref( 0, 0 );
}

void OneToOneMsg::sources( vector< vector< Eref > > & v) const
{
	v.resize( 0 );
	unsigned int n = e1_->numData();
	if ( n > e2_->numData() )
		n = e2_->numData();
	v.resize( e2_->numData() );
	for ( unsigned int i = 0; i < n; ++i ) {
		v[i].resize( 1, Eref( e1_, i ) );
	}
}

void OneToOneMsg::targets( vector< vector< Eref > > & v) const
{
	unsigned int n = e1_->numData();
	if ( n > e2_->numData() )
		n = e2_->numData();
	v.resize( e1_->numData() );
	for ( unsigned int i = 0; i < n; ++i ) {
		v[i].resize( 1, Eref( e2_, i ) );
	}
}

Id OneToOneMsg::managerId() const
{
	return OneToOneMsg::managerId_;
}

ObjId OneToOneMsg::findOtherEnd( ObjId f ) const
{
	if ( f.element() == e1() )
		return ObjId( e2()->id(), f.dataIndex );
	else if ( f.element() == e2() )
		return ObjId( e1()->id(), f.dataIndex );
	
	return ObjId( 0, BADINDEX );
}

Msg* OneToOneMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc.element();
	// This works both for 1-copy and for n-copies
	OneToOneMsg* ret = 0;
	if ( orig == e1() ) {
		ret = new OneToOneMsg( newSrc.element(), newTgt.element(), 0 );
		ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
	} else if ( orig == e2() ) {
		ret = new OneToOneMsg( newTgt.element(), newSrc.element(), 0 );
		ret->e2()->addMsgAndFunc( ret->mid(), fid, b );
	} else
		assert( 0 );
	// ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
	return ret;
}

/// Static function for Msg access
unsigned int OneToOneMsg::numMsg()
{
	return msg_.size();
}

/// Static function for Msg access
char* OneToOneMsg::lookupMsg( unsigned int index )
{
	assert( index < msg_.size() );
	return reinterpret_cast< char* >( msg_[index] );
}

///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* OneToOneMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions. Nothing here.
	///////////////////////////////////////////////////////////////////

	static Dinfo< short > dinfo;
	static Cinfo msgCinfo (
		"OneToOneMsg",	// name
		Msg::initCinfo(),				// base class
		0,								// Finfo array
		0,								// Num Fields
		&dinfo
	);

	return &msgCinfo;
}

static const Cinfo* oneToOneMsgCinfo = OneToOneMsg::initCinfo();

