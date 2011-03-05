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
	// Element Value Finfos
	/////////////////////////////////////////////////////////////////

	static ElementValueFinfo< Neutral, string > name( 
		"name",
		"Name of object", 
		&Neutral::setName, 
		&Neutral::getName );

	static ElementValueFinfo< Neutral, unsigned int > group( 
		"group",
		"Computational group in which object belongs. Assumed to require"
		"high density message traffic, and queues are organized "
		"according to group. Groups are inherited from parent but one" 
		"can select any of the extant groups to use for an Element",
		&Neutral::setGroup, 
		&Neutral::getGroup );

	// Should be renamed to myId
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

	static ElementValueFinfo< Neutral, unsigned int > fieldDimension( 
		"fieldDimension",
		"Max size of the dimension of the array of fields."
		"Applicable specially for ragged arrays of fields, "
		"where each object may have a different number of fields. "
		"Must be larger than the size of any of the ragger arrays."
		"Normally is only assigned from Shell::doSyncDataHandler.",
			&Neutral::setFieldDimension,
			&Neutral::getFieldDimension
		);

	/////////////////////////////////////////////////////////////////
	// Value Finfos
	/////////////////////////////////////////////////////////////////
	static ValueFinfo< Neutral, Neutral > thisFinfo (
		"this",
		"Access function for entire object",
		&Neutral::setThis,
		&Neutral::getThis
	);
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
		&thisFinfo,
		&name,
		&me,
		&parent,
		&children,
		&path,
		&className,
		&fieldDimension,
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

////////////////////////////////////////////////////////////////////////
// Access functions
////////////////////////////////////////////////////////////////////////


void Neutral::setThis( Neutral v )
{
	;
}

Neutral Neutral::getThis() const
{
	return *this;
}

void Neutral::setName( const Eref& e, const Qinfo* q, string name )
{
	e.element()->setName( name );
}

string Neutral::getName( const Eref& e, const Qinfo* q ) const
{
	return e.element()->getName();
}

void Neutral::setGroup( const Eref& e, const Qinfo* q, unsigned int val )
{
	e.element()->setGroup( val );
}

unsigned int Neutral::getGroup( const Eref& e, const Qinfo* q ) const
{
	return e.element()->getGroup();
}

FullId Neutral::getFullId( const Eref& e, const Qinfo* q ) const
{
	return e.fullId();
}

FullId Neutral::getParent( const Eref& e, const Qinfo* q ) const
{
	return parent( e );
	/*
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	MsgId mid = e.element()->findCaller( pafid );
	assert( mid != Msg::badMsg );

	return Msg::getMsg( mid )->findOtherEnd( e.fullId() );
	*/
}

/**
 * Gets Element children, not individual entries in the array.
 */
vector< Id > Neutral::getChildren( const Eref& e, const Qinfo* q ) const
{
	vector< Id > ret;
	children( e, ret );
	return ret;
}

// Static function
void Neutral::children( const Eref& e, vector< Id >& ret )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();
	static const Finfo* cf = neutralCinfo->findFinfo( "childMsg" );
	static const SrcFinfo* cf2 = dynamic_cast< const SrcFinfo* >( cf );
	static const BindIndex bi = cf2->getBindIndex();
	
	const vector< MsgFuncBinding >* bvec = e.element()->getMsgAndFunc( bi );

	for ( vector< MsgFuncBinding >::const_iterator i = bvec->begin();
		i != bvec->end(); ++i ) {
		if ( i->fid == pafid ) {
			const Msg* m = Msg::getMsg( i->mid );
			assert( m );
			ret.push_back( m->e2()->id() );
		}
	}
}

/*
 * Gets specific named child
Id Neutral::getChild( const Eref& e, const Qinfo* q, const string& name ) 
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
*/


string Neutral::getPath( const Eref& e, const Qinfo* q ) const
{
	return path( e );
}

string Neutral::getClass( const Eref& e, const Qinfo* q ) const
{
	return e.element()->cinfo()->name();
}


void Neutral::setFieldDimension( const Eref& e, const Qinfo* q, 
	unsigned int size )
{
	e.element()->dataHandler()->setFieldDimension( size );
}

unsigned int Neutral::getFieldDimension( 
	const Eref& e, const Qinfo* q ) const
{
	return e.element()->dataHandler()->getFieldDimension();
}

unsigned int Neutral::buildTree( const Eref& e, const Qinfo* q, vector< Id >& tree )
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
void Neutral::destroy( const Eref& e, const Qinfo* q, int stage )
{
	vector< Id > tree;
	unsigned int numDescendants = buildTree( e, q, tree );
	/*
	cout << "Neutral::destroy: id = " << e.id() << 
		", name = " << e.element()->getName() <<
		", numDescendants = " << numDescendants << endl;
		*/
	assert( numDescendants == tree.size() );
	Element::destroyElementTree( tree );
}

/////////////////////////////////////////////////////////////////////////
// Static utility functions.
/////////////////////////////////////////////////////////////////////////

// static function
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

// static function
Id Neutral::child( const Eref& e, const string& name ) 
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

// static function
FullId Neutral::parent( const Eref& e )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	if ( e.element()->id() == Id() ) {
		cout << "Warning: Neutral::parent: tried to take parent of root\n";
		return FullId( Id(), 0 );
	}

	MsgId mid = e.element()->findCaller( pafid );
	assert( mid != Msg::badMsg );

	return Msg::getMsg( mid )->findOtherEnd( e.fullId() );
}

// Static function
string Neutral::path( const Eref& e )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	vector< FullId > pathVec;
	FullId curr = e.fullId();
	stringstream ss;

	pathVec.push_back( curr );
	while ( curr.id != Id() ) {
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

// Neutral does not have any fields.
istream& operator >>( istream& s, Neutral& d )
{
	return s;
}

ostream& operator <<( ostream& s, const Neutral& d )
{
	return s;
}
