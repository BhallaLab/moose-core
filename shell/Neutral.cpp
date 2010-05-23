/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Dinfo.h"
#include "ElementValueFinfo.h"


const Cinfo* Neutral::initCinfo()
{
	/////////////////////////////////////////////////////////////////
	// Value Finfos
	/////////////////////////////////////////////////////////////////
	static ElementValueFinfo< Neutral, string > name( 
		"name",
		"Name of object", 
		&Neutral::setName, 
		&Neutral::getName );

	static ReadOnlyElementValueFinfo< Neutral, FullId > parent( 
		"parent",
		"Parent FullId for current object", 
			&Neutral::getParent );

	static ReadOnlyElementValueFinfo< Neutral, string > className( 
		"class",
		"Class Name of object", 
			&Neutral::getClass );
	/////////////////////////////////////////////////////////////////
	// SrcFinfos
	/////////////////////////////////////////////////////////////////
	static SrcFinfo1< int > childMsg( "childMsg", 
		"Message to child Elements");

	/////////////////////////////////////////////////////////////////
	// DestFinfos
	/////////////////////////////////////////////////////////////////
	static DestFinfo parentMsg( "parentMsg", 
		"Message from Parent Element(s)", 
		new EpFunc1< Neutral, int >( &Neutral::destroy ) );
			
	/////////////////////////////////////////////////////////////////
	// Setting up the Finfo list.
	/////////////////////////////////////////////////////////////////
	static Finfo* neutralFinfos[] = {
		&childMsg,
		&parentMsg,
		&name,
		&parent,
		&className,
	};

	/////////////////////////////////////////////////////////////////
	// Setting up the Cinfo.
	/////////////////////////////////////////////////////////////////
	static Cinfo neutralCinfo (
		"Neutral",
		0, // No base class.
		neutralFinfos,
		sizeof( neutralFinfos ) / sizeof( Finfo* ),
		new Dinfo< Neutral >()
	);

	return &neutralCinfo;
}

static const Cinfo* neutralCinfo = Neutral::initCinfo();


Neutral::Neutral()
	// : name_( "" )
{
	;
}

void Neutral::process( const ProcInfo* p, const Eref& e )
{
	;
}

void Neutral::setName( Eref e, const Qinfo* q, string name )
{
	e.element()->setName( name );
}

string Neutral::getName( Eref e, const Qinfo* q ) const
{
	return e.element()->getName();
}

FullId Neutral::getParent( Eref e, const Qinfo* q ) const
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	MsgId mid = e.element()->findCaller( pafid );
	assert( mid != Msg::badMsg );

	return Msg::getMsg( mid )->findOtherEnd( e.fullId() );
}

string Neutral::getClass( Eref e, const Qinfo* q ) const
{
	return e.element()->cinfo()->name();
}

//
// Stage 1: mark for deletion
// Stage 2: Clear out outside-going msgs
// Stage 3: delete self and attached msgs, 
void Neutral::destroy( Eref e, const Qinfo* q, int stage )
{
	// cout << "in Neutral::destroy()[ " << e.index() << "]\n";
	;
}
