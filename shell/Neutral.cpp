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

	static ReadOnlyElementValueFinfo< Neutral, FullId > me( 
		"me",
		"FullId for current object", 
			&Neutral::getFullId );

	static ReadOnlyElementValueFinfo< Neutral, FullId > parent( 
		"parent",
		"Parent FullId for current object", 
			&Neutral::getParent );

	static ReadOnlyElementValueFinfo< Neutral, vector< Id > > children( 
		"children",
		"vector of FullIds listing all children of current object", 
			&Neutral::getChildren );

	static ReadOnlyElementValueFinfo< Neutral, string > path( 
		"path",
		"text path for object", 
			&Neutral::getPath );

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
		&me,
		&parent,
		&children,
		&path,
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

FullId Neutral::getFullId( Eref e, const Qinfo* q ) const
{
	return e.fullId();
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

/**
 * Gets Element children, not individual entries in the array.
 */
vector< Id > Neutral::getChildren( Eref e, const Qinfo* q ) const
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* cf = neutralCinfo->findFinfo( "childMsg" );
	static const SrcFinfo* cf2 = dynamic_cast< const SrcFinfo* >( cf );
	static const BindIndex bi = cf2->getBindIndex();
	
	const vector< MsgFuncBinding >* bvec = e.element()->getMsgAndFunc( bi );

	vector< Id > ret;

	for ( vector< MsgFuncBinding >::const_iterator i = bvec->begin();
		i != bvec->end(); ++i ) {
		if ( i->fid == pafid ) {
			const Msg* m = Msg::getMsg( i->mid );
			assert( m );
			ret.push_back( m->e2()->id() );
		}
	}
	return ret;
}

/**
 * Gets specific named child
 */
Id Neutral::getChild( Eref e, const Qinfo* q, const string& name ) const
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* cf = neutralCinfo->findFinfo( "childMsg" );
	static const SrcFinfo* cf2 = dynamic_cast< const SrcFinfo* >( cf );
	static const BindIndex bi = cf2->getBindIndex();
	
	const vector< MsgFuncBinding >* bvec = e.element()->getMsgAndFunc( bi );

	vector< Id > ret;

	for ( vector< MsgFuncBinding >::const_iterator i = bvec->begin();
		i != bvec->end(); ++i ) {
		if ( i->fid == pafid ) {
			const Msg* m = Msg::getMsg( i->mid );
			assert( m );
			if ( m->e2()->getName() == name )
				return m->e2()->id();
		}
	}
	return Id();
}

string Neutral::getPath( Eref e, const Qinfo* q ) const
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	vector< FullId > pathVec;
	FullId curr = e.fullId();
	stringstream ss;

	pathVec.push_back( curr );
	while ( !( curr == FullId( Id(), 0 ) ) ) {
		MsgId mid = curr.eref().element()->findCaller( pafid );
		assert( mid != Msg::badMsg );
		curr = Msg::getMsg( mid )->findOtherEnd( curr );
		pathVec.push_back( curr );
	}

	ss << "/";
	for ( int i = pathVec.size() - 2; i >= 0; --i ) {
		ss << pathVec[i].eref();
		if ( i > 0 )
			ss << "/";
	}
	return ss.str();
}

string Neutral::getClass( Eref e, const Qinfo* q ) const
{
	return e.element()->cinfo()->name();
}

unsigned int Neutral::buildTree( Eref e, const Qinfo* q, vector< Id >& tree )
	const 
{
	unsigned int ret = 1;
	tree.push_back( e.element()->id() );
	vector< Id > kids = getChildren( e, q );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
		ret += buildTree( i->eref(), q, tree );
	return ret;
}

//
// Stage 1: mark for deletion. This is done by setting cinfo = 0
// Stage 2: Clear out outside-going msgs
// Stage 3: delete self and attached msgs, 
void Neutral::destroy( Eref e, const Qinfo* q, int stage )
{
	vector< Id > tree;
	unsigned int numDescendants = buildTree( e, q, tree );
	assert( numDescendants == tree.size() );
	Element::destroyElementTree( tree );
}

bool Neutral::isDescendant( Id me, Id ancestor )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	Eref e = me.eref();
	
	while ( e.element()->id() != Id() && e.element()->id() != ancestor ) {
		MsgId mid = e.element()->findCaller( pafid );
		assert( mid != Msg::badMsg );
		FullId fid = Msg::getMsg( mid )->findOtherEnd( e.fullId() );
		e = fid.eref();
	}
	return ( e.element()->id() == ancestor );
}
