/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OneToAllMsg.h"

Id OneToAllMsg::managerId_;

OneToAllMsg::OneToAllMsg( MsgId mid, Eref e1, Element* e2 )
	: 
		Msg( mid, e1.element(), e2, OneToAllMsg::managerId_ ),
		i1_( e1.dataIndex() )
{
	;
}

OneToAllMsg::~OneToAllMsg()
{
	// I cannot do this in the Msg::~Msg destructor because the virtual
	// functions  for managerId() don't work there.
	destroyDerivedMsg( managerId_, mid_ );
}

Eref OneToAllMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ )
		return Eref( e2_, 0 );
	else if ( src.element() == e2_ )
		return Eref( e1_, i1_ );
	return Eref( 0, 0 );
}

void OneToAllMsg::sources( vector< vector < Eref > >& v ) const
{
	// Same single source for all targets.
	v.clear();
	vector< Eref > temp( 1, Eref( e1_, i1_ ) );
	v.assign( e2_->numData(), temp  );
}

void OneToAllMsg::targets( vector< vector< Eref > >& v ) const
{
	v.clear();
	v.resize( e1_->numData() );
	v[i1_].resize( 1, Eref( e2_, ALLDATA ) );
}

Id OneToAllMsg::managerId() const
{
	return OneToAllMsg::managerId_;
}

ObjId OneToAllMsg::findOtherEnd( ObjId f ) const
{
	if ( f.element() == e1() ) {
		if ( f.dataId == i1_ )
			return ObjId( e2()->id(), 0 );
		else
		  return ObjId( Id() );
	} else if ( f.element() == e2() ) {
		return ObjId( e1()->id(), i1_ );
	}
	
	return ObjId::bad();
}

Msg* OneToAllMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc.element();
	if ( n <= 1 ) {
		OneToAllMsg* ret = 0;
		if ( orig == e1() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), 
					Eref( newSrc.element(), i1_ ), newTgt.element() );
			ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
		} else if ( orig == e2() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), 
					Eref( newTgt.element(), i1_ ), newSrc.element() );
			ret->e2()->addMsgAndFunc( ret->mid(), fid, b );
		} else {
			assert( 0 );
		}
		return ret;
	} else {
		// Here we need a SliceMsg which goes from one 2-d array to another.
		cout << "Error: OneToAllMsg::copy: SliceToSliceMsg not yet implemented\n";
		return 0;
	}
}

///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* OneToAllMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< OneToAllMsg, DataId > i1(
		"i1",
		"DataId of source Element.",
		&OneToAllMsg::getI1
	);

	static Finfo* msgFinfos[] = {
		&i1,		// readonly value
	};

	static Dinfo< short > dinfo;
	static Cinfo msgCinfo (
		"OneToAllMsg",	// name
		Msg::initCinfo(),				// base class
		msgFinfos,
		sizeof( msgFinfos ) / sizeof( Finfo* ),	// num Fields
		&dinfo
	);

	return &msgCinfo;
}

static const Cinfo* assignmentMsgCinfo = OneToAllMsg::initCinfo();

/**
 * Return the first DataId
 */
DataId OneToAllMsg::getI1() const
{
	return i1_;
}
