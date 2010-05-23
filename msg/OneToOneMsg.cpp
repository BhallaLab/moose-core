/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgManager.h"
#include "OneToOneMsg.h"

Id OneToOneMsg::id_;

const Cinfo* OneToOneMsgWrapper::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< OneToOneMsgWrapper, Id > element1(
		"e1",
		"Id of source Element.",
		&OneToOneMsgWrapper::getE1
	);
	static ReadOnlyValueFinfo< OneToOneMsgWrapper, Id > element2(
		"e2",
		"Id of source Element.",
		&OneToOneMsgWrapper::getE2
	);

	static Finfo* oneToOneMsgFinfos[] = {
		&element1,		// readonly value
		&element2,		// readonly value
	};

	static Cinfo oneToOneMsgCinfo (
		"OneToOneMsg",					// name
		MsgManager::initCinfo(),		// base class
		oneToOneMsgFinfos,
		sizeof( oneToOneMsgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< OneToOneMsgWrapper >()
	);

	return &oneToOneMsgCinfo;
}

static const Cinfo* oneToOneMsgCinfo = OneToOneMsgWrapper::initCinfo();

//////////////////////////////////////////////////////////////////////

OneToOneMsg::OneToOneMsg( Element* e1, Element* e2 )
	: Msg( e1, e2, id_ )
{
	;
}

OneToOneMsg::~OneToOneMsg()
{
	MsgManager::dropMsg( mid() );
}

void OneToOneMsg::exec( const char* arg, const ProcInfo* p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	unsigned int src = q->srcIndex().data(); // will also be dest index.
	if ( q->isForward() ) {
		if ( e2_->dataHandler()->isDataHere( src ) ) {
			const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e2_, q->srcIndex() ), arg );
		}
	} else {
		if ( e1_->dataHandler()->isDataHere( src ) ) {
			const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e1_, q->srcIndex() ), arg );
		}
	}
}

Id OneToOneMsg::id() const
{
	return id_;
}

FullId OneToOneMsg::findOtherEnd( FullId f ) const
{
	if ( f.id() == e1() )
		return FullId( e2()->id(), f.dataId );
	else if ( f.id() == e2() )
		return FullId( e1()->id(), f.dataId );
	
	return FullId::bad();
}
