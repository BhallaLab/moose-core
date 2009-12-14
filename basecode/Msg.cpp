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
const MsgId Msg::Null = 0;

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

/**
 * Here it has to slot the data into the appropriate queue, depending on
 * which Elements and objects reside on which thread.
 * In other words, the Msg needs additional info in cases where we handle
 * multiple threads. The current 
 */
void Msg::addToQ( const Element* caller, Qinfo& q, 
	const ProcInfo* p, const char* arg ) const
{
	// The base function just bungs the data into the one and only queue.
	q.addToQ( 0, mid_, ( caller == e1_ ), arg );

/*
	q.setForward( caller == e1_ );
	q.setMsgId( mid_ );
	if ( caller == e1_ ) {
		q.setMsgId( m2_ );
		e2_->addToQ( q, arg );
	} else {
		assert( caller == e2_ );
		q.setMsgId( m1_ );
		e1_->addToQ( q, arg );
	}
	*/
}

const Msg* Msg::getMsg( MsgId m )
{
	assert( m < msg_.size() );
	return msg_[ m ];
}

///////////////////////////////////////////////////////////////////////////

SingleMsg::SingleMsg( Eref e1, Eref e2 )
	: Msg( e1.element(), e2.element() ),
	i1_( e1.index() ), 
	i2_( e2.index() )
{
	;
}

void SingleMsg::exec( const char* arg ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e2_, i2_ ), arg );
	} else {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

/*
void SingleMsg::exec( Element* target, const char* arg ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	// arg += sizeof( Qinfo );
	const OpFunc* f = target->cinfo()->getOpFunc( q->fid() );
	if ( target == e1_ ) {
		f->op( Eref( target, i1_ ), arg );
	} else {
		assert( target == e2_ );
		f->op( Eref( target, i2_ ), arg );
	}
}
*/

bool SingleMsg::add( Eref e1, const string& srcField, 
			Eref e2, const string& destField )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1.element(), srcField,
		e2.element(), destField, funcId );

	if ( srcFinfo ) {
		Msg* m = new SingleMsg( e1, e2 );
		e1.element()->addMsgToConn( m->mid(), srcFinfo->getConnId() );
		e1.element()->addTargetFunc( funcId, srcFinfo->getFuncIndex() );
		return 1;
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////

OneToOneMsg::OneToOneMsg( Element* e1, Element* e2 )
	: Msg( e1, e2 )
{
	;
}

void OneToOneMsg::exec( const char* arg ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e2_, q->srcIndex() ), arg );
	} else {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, q->srcIndex() ), arg );
	}
}


///////////////////////////////////////////////////////////////////////////

OneToAllMsg::OneToAllMsg( Eref e1, Element* e2 )
	: 
		Msg( e1.element(), e2 ),
		i1_( e1.index() )
{
	;
}

void OneToAllMsg::exec( const char* arg ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	// arg += sizeof( Qinfo );
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		if ( e2_->numDimensions() == 1 ) {
			for ( unsigned int i = 0; i < e2_->numData(); ++i )
				f->op( Eref( e2_, i ), arg );
		} else if ( e2_->numDimensions() == 2 ) {
			for ( unsigned int i = 0; i < e2_->numData1(); ++i )
				for ( unsigned int j = 0; j < e2_->numData2( i ); ++j )
					f->op( Eref( e2_, DataId( i, j ) ), arg );
		}
	} else {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

bool OneToAllMsg::add( Eref e1, const string& srcField, 
			Element* e2, const string& destField )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1.element(), srcField,
		e2, destField, funcId );

	if ( srcFinfo ) {
		Msg* m = new OneToAllMsg( e1, e2 );
		e1.element()->addMsgToConn( m->mid(), srcFinfo->getConnId() );
		e1.element()->addTargetFunc( funcId, srcFinfo->getFuncIndex() );
		return 1;
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////
