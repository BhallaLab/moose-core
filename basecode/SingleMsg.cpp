/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SingleMsg.h"
#include "Message.h"

const Cinfo* SingleMsgWrapper::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< SingleMsgWrapper, Id > element1(
		"e1",
		"Id of source Element.",
		&SingleMsgWrapper::getE1
	);
	static ReadOnlyValueFinfo< SingleMsgWrapper, Id > element2(
		"e2",
		"Id of source Element.",
		&SingleMsgWrapper::getE2
	);
	static ValueFinfo< SingleMsgWrapper, DataId > index1(
		"i1",
		"Index of source object.",
		&SingleMsgWrapper::setI1,
		&SingleMsgWrapper::getI1
	);
	static ValueFinfo< SingleMsgWrapper, DataId > index2(
		"i2",
		"Index of dest object.",
		&SingleMsgWrapper::setI2,
		&SingleMsgWrapper::getI2
	);

	static Finfo* singleMsgFinfos[] = {
		&element1,		// readonly value
		&element2,		// readonly value
		&index1,		// value
		&index2,		// value
	};

	static Cinfo singleMsgCinfo (
		"SingleMsg",					// name
		MsgManager::initCinfo(),		// base class
		singleMsgFinfos,
		sizeof( singleMsgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< SingleMsgWrapper >()
	);

	return &singleMsgCinfo;
}

static const Cinfo* singleMsgCinfo = SingleMsgWrapper::initCinfo();

/*
Id SingleMsgWrapper::getE1() const
{
	const Msg* m = Msg::safeGetMsg( mid_ );
	if ( m ) {
		return m->e1()->id();
	}
	return Id();
}

Id SingleMsgWrapper::getE2() const
{
	const Msg* m = Msg::safeGetMsg( mid_ );
	if ( m ) {
		return m->e2()->id();
	}
	return Id();
}
*/

DataId SingleMsgWrapper::getI1() const
{
	const Msg* m = Msg::safeGetMsg( getMid() );
	if ( m ) {
		const SingleMsg* sm = dynamic_cast< const SingleMsg* >( m );
		if ( sm ) {
			return sm->i1();
		}
	}
	return DataId( 0 );
}

void SingleMsgWrapper::setI1( DataId di )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	if ( m ) {
		SingleMsg* sm = dynamic_cast< SingleMsg* >( m );
		if ( sm ) {
			sm->setI1( di );
		}
	}
}

DataId SingleMsgWrapper::getI2() const
{
	const Msg* m = Msg::safeGetMsg( getMid() );
	if ( m ) {
		const SingleMsg* sm = dynamic_cast< const SingleMsg* >( m );
		if ( sm ) {
			return sm->i2();
		}
	}
	return DataId( 0 );
}

void SingleMsgWrapper::setI2( DataId di )
{
	Msg* m = Msg::safeGetMsg( getMid() );
	if ( m ) {
		SingleMsg* sm = dynamic_cast< SingleMsg* >( m );
		if ( sm ) {
			sm->setI2( di );
		}
	}
}

/////////////////////////////////////////////////////////////////////
// Here is the SingleMsg code
/////////////////////////////////////////////////////////////////////

SingleMsg::SingleMsg( Eref e1, Eref e2 )
	: Msg( e1.element(), e2.element(), id_ ),
	i1_( e1.index() ), 
	i2_( e2.index() )
{
	;
}

SingleMsg::~SingleMsg()
{
	MsgManager::dropMsg( mid() );
}

void SingleMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() && e2_->dataHandler()->isDataHere( i2_ ) ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e2_, i2_ ), arg );
		return;
	} 

	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

bool SingleMsg::isMsgHere( const Qinfo& q ) const
{
	if ( q.isForward() )
		return ( i1_ == q.srcIndex() );
	else
		return ( i2_ == q.srcIndex() );
}

DataId SingleMsg::i1() const
{
	return i1_;
}

DataId SingleMsg::i2() const
{
	return i2_;
}

void SingleMsg::setI1( DataId di )
{
	i1_ = di;
}

void SingleMsg::setI2( DataId di )
{
	i2_ = di;
}

Id SingleMsg::id() const 
{
	return id_;
}

/*
// Deprecated.
bool SingleMsg::add( Eref e1, const string& srcField, 
			Eref e2, const string& destField )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1.element(), srcField,
		e2.element(), destField, funcId );

	if ( srcFinfo ) {
		Msg* m = new SingleMsg( e1, e2 );
		e1.element()->addMsgAndFunc( m->mid(), funcId, srcFinfo->getBindIndex() );
		return 1;
	}
	return 0;
}
*/

/// Static function used during initialization
void SingleMsg::setId( Id id )
{
	id_ = id;
}
