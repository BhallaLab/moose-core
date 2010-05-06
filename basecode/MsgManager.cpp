/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

const Cinfo* MsgManager::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< MsgManager, Id > element1(
		"e1",
		"Id of source Element.",
		&MsgManager::getE1
	);
	static ReadOnlyValueFinfo< MsgManager, Id > element2(
		"e2",
		"Id of source Element.",
		&MsgManager::getE2
	);

	static Finfo* msgManagerFinfos[] = {
		&element1,		// readonly value
		&element2,		// readonly value
	};

	static Cinfo msgManagerCinfo (
		"MsgManager",	// name
		0,				// base class
		msgManagerFinfos,
		sizeof( msgManagerFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< MsgManager >()
	);

	return &msgManagerCinfo;
}

static const Cinfo* msgManagerCinfo = MsgManager::initCinfo();

//////////////////////////////////////////////////////////////////////
//  Now into the class functions
//////////////////////////////////////////////////////////////////////

MsgManager::MsgManager()
	: mid_( 0 )
{;}

MsgManager::MsgManager( MsgId mid )
	: mid_( mid )
{;}

Id MsgManager::getE1() const
{
	const Msg* m = Msg::safeGetMsg( mid_ );
	if ( m ) {
		return m->e1()->id();
	}
	return Id();
}

Id MsgManager::getE2() const
{
	const Msg* m = Msg::safeGetMsg( mid_ );
	if ( m ) {
		return m->e2()->id();
	}
	return Id();
}

void MsgManager::setMid( MsgId mid )
{
	mid_ = mid;
}

MsgId MsgManager::getMid() const
{
	return mid_;
}

// static func
void MsgManager::addMsg( MsgId mid, Id managerId )
{
	const Msg* m = Msg::getMsg( mid );
	Eref manager = m->manager( managerId );
	Element* em = manager.element();
	DataHandler* data = em->dataHandler();
	MsgManager mm( mid );
	unsigned int nextDataId = data->addOneEntry( 
		reinterpret_cast< const char* >( &mm ) );
	m->setDataId( nextDataId );
}

// static func
void MsgManager::dropMsg( MsgId mid )
{
	const Msg* m = Msg::getMsg( mid );
	Eref manager = m->manager( m->id() );
	MsgManager* mm = reinterpret_cast< MsgManager* >( manager.data() );
	mm->setMid( 0 );
	m->setDataId( 0 );
}
