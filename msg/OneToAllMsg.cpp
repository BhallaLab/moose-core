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
		i1_( e1.index() )
{
	;
}

OneToAllMsg::~OneToAllMsg()
{
	// I cannot do this in the Msg::~Msg destructor because the virtual
	// functions  for managerId() don't work there.
	destroyDerivedMsg( managerId_, mid_ );
	/*
	cout << "Deleting OneToAllMsg from " << e1_->getName() << " to " <<
		e2_->getName() << endl;
	// Here we have a special case: The parent-child messages are
	// OneToAll. So when cleaning up the whole simulation, it removes the
	// managerId_.element() while there are still some parent-child
	// messages present. For protection, don't do the deletion if the
	// element has gone.
	if ( managerId_.element() ) {
		MsgDataHandler * mdh = dynamic_cast< MsgDataHandler* >( 
			managerId_.element()->dataHandler() );
		assert( mdh );
		mdh->dropMid( mid_ );
	}
		*/
}


Eref OneToAllMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ )
		return Eref( e2_, 0 );
	else if ( src.element() == e2_ )
		return Eref( e1_, i1_ );
	return Eref( 0, 0 );
}

Id OneToAllMsg::managerId() const
{
	return OneToAllMsg::managerId_;
}

ObjId OneToAllMsg::findOtherEnd( ObjId f ) const
{
	if ( f.id() == e1() ) {
		if ( f.dataId == i1_ )
			return ObjId( e2()->id(), 0 );
		else
		  return ObjId( Id() );
	} else if ( f.id() == e2() ) {
		return ObjId( e1()->id(), i1_ );
	}
	
	return ObjId::bad();
}

Msg* OneToAllMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc();
	if ( n <= 1 ) {
		OneToAllMsg* ret = 0;
		if ( orig == e1() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), Eref( newSrc(), i1_ ), newTgt() );
			ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
		} else if ( orig == e2() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), Eref( newTgt(), i1_ ), newSrc() );
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

	static Cinfo msgCinfo (
		"OneToAllMsg",	// name
		Msg::initCinfo(),				// base class
		msgFinfos,
		sizeof( msgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< short >()
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
