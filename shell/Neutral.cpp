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
#include "LookupElementValueFinfo.h"

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
	static ReadOnlyElementValueFinfo< Neutral, ObjId > me( 
		"me",
		"ObjId for current object", 
			&Neutral::getObjId );

	static ReadOnlyElementValueFinfo< Neutral, ObjId > parent( 
		"parent",
		"Parent ObjId for current object", 
			&Neutral::getParent );

	static ReadOnlyElementValueFinfo< Neutral, vector< Id > > children( 
		"children",
		"vector of ObjIds listing all children of current object", 
			&Neutral::getChildren );

	static ReadOnlyElementValueFinfo< Neutral, string > path( 
		"path",
		"text path for object", 
			&Neutral::getPath );

	static ReadOnlyElementValueFinfo< Neutral, string > className( 
		"class",
		"Class Name of object", 
			&Neutral::getClass );

	static ReadOnlyElementValueFinfo< Neutral, unsigned int > linearSize( 
		"linearSize",
		"# of entries on Element: product of all dimensions."
		"Note that on a FieldElement this includes field entries."
		"If field entries form a ragged array, then the linearSize may be"
		"greater than the actual number of allocated entries, since the"
		"fieldDimension is at least as big as the largest ragged array.",
			&Neutral::getLinearSize );

	static ReadOnlyElementValueFinfo< Neutral, vector< unsigned int > > dimensions( 
		"dimensions",
		"Dimensions of data on the Element." 
		"This includes the fieldDimension if present.",
			&Neutral::getDimensions );

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

	static ReadOnlyElementValueFinfo< Neutral, unsigned int > 
		localNumField( 
		"localNumField",
		"For a FieldElement: number of entries of self on current node"
		"For a regular Element: zero.",
			&Neutral::getLocalNumField );

	static ReadOnlyElementValueFinfo< Neutral, vector< ObjId > > msgOut( 
		"msgOut",
		"Messages going out from this Element", 
			&Neutral::getOutgoingMsgs );

	static ReadOnlyElementValueFinfo< Neutral, vector< ObjId > > msgIn( 
		"msgIn",
		"Messages coming in to this Element", 
			&Neutral::getIncomingMsgs );

	static ReadOnlyLookupElementValueFinfo< Neutral, string, vector< Id > > msgSrc( 
		"msgSrc",
		"Source Ids of Messages coming into this Element on specified field", 
			&Neutral::getMsgSourceIds );

	static ReadOnlyLookupElementValueFinfo< Neutral, string, vector< Id > > msgDest( 
		"msgDest",
		"Destination Ids of Messages from this Element on specified field", 
			&Neutral::getMsgTargetIds );

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
		&linearSize,
		&dimensions,
		&fieldDimension,
		&localNumField,
		&msgOut,
		&msgIn,
		&msgSrc,
		&msgDest,
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

ObjId Neutral::getObjId( const Eref& e, const Qinfo* q ) const
{
	return e.objId();
}

ObjId Neutral::getParent( const Eref& e, const Qinfo* q ) const
{
	return parent( e );
	/*
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	MsgId mid = e.element()->findCaller( pafid );
	assert( mid != Msg::badMsg );

	return Msg::getMsg( mid )->findOtherEnd( e.objId() );
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

unsigned int Neutral::getLinearSize( const Eref& e, const Qinfo* q ) const
{
	return e.element()->dataHandler()->totalEntries();
}

vector< unsigned int > Neutral::getDimensions( 
	const Eref& e, const Qinfo* q ) const
{
	return e.element()->dataHandler()->dims();
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

unsigned int Neutral::getLocalNumField( 
	const Eref& e, const Qinfo* q ) const
{
	return e.element()->dataHandler()->getFieldArraySize( e.index().data());
}

vector< ObjId > Neutral::getOutgoingMsgs( 
	const Eref& e, const Qinfo* q ) const
{
	vector< ObjId > ret;
	unsigned int numBindIndex = e.element()->cinfo()->numBindIndex();

	for ( unsigned int i = 0; i < numBindIndex; ++i ) {
		const vector< MsgFuncBinding >* v = 
			e.element()->getMsgAndFunc( i );
		if ( v ) {
			for ( vector< MsgFuncBinding >::const_iterator mb = v->begin();
				mb != v->end(); ++mb ) {
				const Msg* m = Msg::getMsg( mb->mid );
				assert( m );
				ret.push_back( m->manager().objId() );
			}
		}
	}
	return ret;
}

vector< ObjId > Neutral::getIncomingMsgs( 
	const Eref& e, const Qinfo* q ) const
{
	vector< ObjId > ret;
	const vector< MsgId >& msgIn = e.element()->msgIn();

	for (unsigned int i = 0; i < msgIn.size(); ++i ) {
		const Msg* m = Msg::getMsg( msgIn[i] );
			assert( m );
			if ( m->e2() == e.element() && m->mid() != Msg::setMsg )
				ret.push_back( m->manager().objId() );
	}
	return ret;
}

vector< Id > Neutral::getMsgTargetIds( 
	const Eref& e, const Qinfo* q, string field ) const
{
	vector< Id > ret;
	const Finfo* f = e.element()->cinfo()->findFinfo( field );
	const SrcFinfo* srcFinfo = dynamic_cast< const SrcFinfo* >( f );
	if ( srcFinfo ) {
		e.element()->getOutputs( ret, srcFinfo );
	}
	return ret;
}


vector< Id > Neutral::getMsgSourceIds( 
	const Eref& e, const Qinfo* q, string field ) const
{
	vector< Id > ret;
	const Finfo* f = e.element()->cinfo()->findFinfo( field );
	const DestFinfo* destFinfo = dynamic_cast< const DestFinfo* >( f );
	if ( destFinfo ) {
		e.element()->getInputs( ret, destFinfo );
	}
	return ret;
}

//////////////////////////////////////////////////////////////////////////

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
		ObjId fid = Msg::getMsg( mid )->findOtherEnd( e.objId() );
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
ObjId Neutral::parent( const Eref& e )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	if ( e.element()->id() == Id() ) {
		cout << "Warning: Neutral::parent: tried to take parent of root\n";
		return ObjId( Id(), 0 );
	}

	MsgId mid = e.element()->findCaller( pafid );
	assert( mid != Msg::badMsg );

	return Msg::getMsg( mid )->findOtherEnd( e.objId() );
}

// Static function
string Neutral::path( const Eref& e )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	vector< ObjId > pathVec;
	ObjId curr = e.objId();
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

bool Neutral::isGlobalField( const string& field )
{
	/*
	* This is over the top: only 3 cases to worry about.
	static set< string > fieldnames;

	if ( fieldnames.size() == 0 ) {
		fieldnames.insert( "set_name" );
		fieldnames.insert( "set_group" );
		fieldnames.insert( "set_fieldDimension" );
	}
	*/
	if ( field.length() < 8 )
		return 0;
	if ( field.substr( 0, 4 ) == "set_" ) {
		if ( field == "set_name" )
			return 1;
		if ( field == "set_group" )
			return 1;
		if ( field == "set_fieldDimension" ) // This is the critical one!
			return 1;
	}
	return 0;
}

