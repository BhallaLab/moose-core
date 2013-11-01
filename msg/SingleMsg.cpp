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
	destroyDerivedMsg( managerId_, mid_ );
}

Eref SingleMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ )
		return Eref( e2_, i2_ );
	else if ( src.element() == e2_ )
		return Eref( e1_, i1_ );
	return Eref( 0, 0 );
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
		  return ObjId( Id() );
	}
	else if ( f.id() == e2() ) {
		if ( f.dataId == i2_ )
			return ObjId( e1()->id(), i1_ );
		else
		  return ObjId( Id() );
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
