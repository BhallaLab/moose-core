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

Id SingleMsg::managerId_;

/////////////////////////////////////////////////////////////////////
// Here is the SingleMsg code
/////////////////////////////////////////////////////////////////////

SingleMsg::SingleMsg( MsgId mid, Eref e1, Eref e2 )
	: Msg( mid, e1.element(), e2.element(), SingleMsg::managerId_ ),
	i1_( e1.index() ), 
	i2_( e2.index() )
{
	;
}

SingleMsg::~SingleMsg()
{
	;
}

void SingleMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	// cout <<  p->nodeIndexInGroup << "." << p->threadIndexInGroup << ": " << e2_->getName() << ", " << i2_;
	/// This partitions the messages between threads.
	if ( q->isForward() && e2_->dataHandler()->isDataHere( i2_ ) &&
		p->execThread( e2_->id(), i2_.data() ) )
	{
			const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
			//cout << ": called\n";
			f->op( Eref( e2_, i2_ ), arg );
			return;
	} 
		// cout << ": NOT called\n";
	
	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) &&
		p->execThread( e1_->id(), i1_.data() ) )
	{
			const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e1_, i1_ ), arg );
	}
}

/*
bool SingleMsg::isMsgHere( const Qinfo& q ) const
{
	if ( q.isForward() )
		return ( i1_ == q.srcIndex() );
	else
		return ( i2_ == q.srcIndex() );
}
*/

DataId SingleMsg::i1() const
{
	return i1_;
}

DataId SingleMsg::i2() const
{
	return i2_;
}

Id SingleMsg::managerId() const 
{
	return SingleMsg::managerId_;
}

ObjId SingleMsg::findOtherEnd( ObjId f ) const
{
	if ( f.id() == e1() ) {
		if ( f.dataId == i1_ )
			return ObjId( e2()->id(), i2_ );
		else
			return ObjId( e2()->id(), DataId::bad() );
	}
	else if ( f.id() == e2() ) {
		if ( f.dataId == i2_ )
			return ObjId( e1()->id(), i1_ );
		else
			return ObjId( e1()->id(), DataId::bad() );
	}
	
	return ObjId::bad();
}

Msg* SingleMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc();
	if ( n <= 1 ) {
		SingleMsg* ret = 0;
		if ( orig == e1() ) {
			ret = new SingleMsg( Msg::nextMsgId(), Eref( newSrc(), i1_ ), Eref( newTgt(), i2_ ) );
			ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
		} else if ( orig == e2() ) {
			ret = new SingleMsg( Msg::nextMsgId(), Eref( newTgt(), i1_ ), Eref( newSrc(), i2_ ) );
			ret->e2()->addMsgAndFunc( ret->mid(), fid, b );
		} else {
			assert( 0 );
		}
		return ret;
	} else {
		// Here we need a SliceMsg which goes from one 2-d array to another.
		cout << "Error: SingleMsg::copy: SliceMsg not yet implemented\n";
		return 0;
	}
}

void SingleMsg::addToQ( const Element* src, Qinfo& q,
	const ProcInfo* p, MsgFuncBinding i, const char* arg ) const
{
	if ( e1_ == src && i1_ == q.srcIndex() ) {
		q.addToQforward( p, i, arg ); // 1 for isForward
	} else if ( e2_ == src && i2_ == q.srcIndex() ) {
		q.addToQbackward( p, i, arg ); 
	}
}


///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* SingleMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ValueFinfo< SingleMsg, DataId > index1(
		"i1",
		"Index of source object.",
		&SingleMsg::setI1,
		&SingleMsg::getI1
	);
	static ValueFinfo< SingleMsg, DataId > index2(
		"i2",
		"Index of dest object.",
		&SingleMsg::setI2,
		&SingleMsg::getI2
	);

	static Finfo* singleMsgFinfos[] = {
		&index1,		// value
		&index2,		// value
	};

	static Cinfo singleMsgCinfo (
		"SingleMsg",					// name
		Msg::initCinfo(),		// base class
		singleMsgFinfos,
		sizeof( singleMsgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< short >()
	);

	return &singleMsgCinfo;
}

static const Cinfo* singleMsgCinfo = SingleMsg::initCinfo();


DataId SingleMsg::getI1() const
{
	return i1_;
}

void SingleMsg::setI1( DataId di )
{
	i1_ = di;
}

DataId SingleMsg::getI2() const
{
	return i2_;
}

void SingleMsg::setI2( DataId di )
{
	i2_ = di;
}
