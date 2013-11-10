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
#include "Shell.h"

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

	/*
			*/
	static ReadOnlyElementValueFinfo< Neutral, string > className( 
		"className",
		"Class Name of object", 
			&Neutral::getClass );

	static ElementValueFinfo< Neutral, unsigned int > numData( 
		"numData",
		"# of entries on Element: product of all dimensions."
		"Note that on a FieldElement this includes field entries."
		"If field entries form a ragged array, then the linearSize may be"
		"greater than the actual number of allocated entries, since the"
		"lastDimension is at least as big as the largest ragged array.",
			&Neutral::setNumData,
			&Neutral::getNumData );

	static ElementValueFinfo< Neutral, unsigned int > 
		numField( 
		"numField",
		"For a FieldElement: number of entries of self"
		"For a regular Element: zero.",
			&Neutral::setNumField,
			&Neutral::getNumField );

	static ReadOnlyElementValueFinfo< Neutral, vector< ObjId > > msgOut( 
		"msgOut",
		"Messages going out from this Element", 
			&Neutral::getOutgoingMsgs );

	static ReadOnlyElementValueFinfo< Neutral, vector< ObjId > > msgIn( 
		"msgIn",
		"Messages coming in to this Element", 
			&Neutral::getIncomingMsgs );

	static ReadOnlyLookupElementValueFinfo< Neutral, string, vector< Id > > neighbours( 
		"neighbours",
		"Ids of Elements connected this Element on specified field.", 
			&Neutral::getNeighbours );

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

	static DestFinfo blockNodeBalance( "blockNodeBalance",
		"Request conversion of data into a blockDataHandler subclass,"
		"and to carry out node balancing of data."
		"The amount of data is not altered, and if the data is already a"
		"blockDataHandler subclass it retains its original dimensions"
		"If converting from a generalDataHandler to blockHandler it treats"
		"it as a linear array"
		"The function preserves the old data and shuffles info around"
		"between nodes to complete the balancing."
		"Arguments are: myNode, startNode, endNode",
		new EpFunc3< Neutral, unsigned int, unsigned int, unsigned int >(
			&Neutral::blockNodeBalance ) );

	static DestFinfo generalNodeBalance( "generalNodeBalance",
		"Request conversion of data into a generalDataHandler subclass,"
		"and to carry out node balancing of data as per specified args."
		"The amount of data is not altered. The generalDataHandler loses"
		"any dimensional information that may have been present in an "
		"earlier blockDataHandler incarnation."
		"Arguments are: myNode, nodeAssignment",
		new EpFunc2< Neutral, unsigned int, vector< unsigned int > >(
			&Neutral::generalNodeBalance ) );
			
	/////////////////////////////////////////////////////////////////
	// Setting up the Finfo list.
	/////////////////////////////////////////////////////////////////
	static Finfo* neutralFinfos[] = {
		&childMsg,				// SrcFinfo
		&parentMsg,				// DestFinfo
		&thisFinfo,				// Value
		&name,					// Value
		&me,					// ReadOnlyValue
		&parent,				// ReadOnlyValue
		&children,				// ReadOnlyValue
		&path,					// ReadOnlyValue
		&className,				// ReadOnlyValue
		&numData,				// Value
		&numField,				// Value
		&msgOut,				// ReadOnlyValue
		&msgIn,					// ReadOnlyValue
		&neighbours,			// ReadOnlyLookupValue
	};

	/////////////////////////////////////////////////////////////////
	// Setting up the Cinfo.
	/////////////////////////////////////////////////////////////////

	static string doc[] =
	{
		"Name", "Neutral",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Neutral: Base class for all MOOSE classes. Provides"
		"access functions for housekeeping fields and operations, message"
		"traversal, and so on."
	};
	static Dinfo< Neutral > dinfo;
	static Cinfo neutralCinfo (
		"Neutral",
		0, // No base class.
		neutralFinfos,
		sizeof( neutralFinfos ) / sizeof( Finfo* ),
		&dinfo,
		doc, 6
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

void Neutral::setName( const Eref& e, string name )
{
	e.element()->setName( name );
}

string Neutral::getName( const Eref& e ) const
{
	return e.element()->getName();
}

ObjId Neutral::getObjId( const Eref& e ) const
{
	return e.objId();
}

ObjId Neutral::getParent( const Eref& e ) const
{
	return parent( e );
}

/**
 * Gets Element children, not individual entries in the array.
 */
vector< Id > Neutral::getChildren( const Eref& e ) const
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


string Neutral::getPath( const Eref& e ) const
{
	return path( e );
}

string Neutral::getClass( const Eref& e ) const
{
	return e.element()->cinfo()->name();
}

unsigned int Neutral::getNumData( const Eref& e ) const
{
	return e.element()->numData();
}

void Neutral::setNumData( const Eref& e, unsigned int num )
{
	e.element()->resize( num );
}

unsigned int Neutral::getNumField( const Eref& e ) const
{
	return e.element()->numField( e.dataIndex() );
}

void Neutral::setNumField( const Eref& e, unsigned int num )
{
	e.element()->resizeField( e.dataIndex(), num );
}

vector< ObjId > Neutral::getOutgoingMsgs( const Eref& e ) const
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

vector< ObjId > Neutral::getIncomingMsgs( const Eref& e ) const
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

vector< Id > Neutral::getNeighbours( const Eref& e, string field ) const
{
	vector< Id > ret;
	const Finfo* finfo = e.element()->cinfo()->findFinfo( field );
	if ( finfo )
		e.element()->getNeighbours( ret, finfo );
	else
		cout << "Warning: Neutral::getNeighbours: Id.Field '" << 
				e.id().path() << "." << field <<
				"' not found\n";
	return ret;
}

//////////////////////////////////////////////////////////////////////////

unsigned int Neutral::buildTree( const Eref& e, vector< Id >& tree )
	const 
{
	unsigned int ret = 1;
	vector< Id > kids = getChildren( e );
	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i )
		ret += buildTree( i->eref(), tree );
	tree.push_back( e.element()->id() );
	return ret;
}
//////////////////////////////////////////////////////////////////////////
// Dest Functions
//////////////////////////////////////////////////////////////////////////

//
// Stage 1: mark for deletion. This is done by setting cinfo = 0
// Stage 2: Clear out outside-going msgs
// Stage 3: delete self and attached msgs, 
void Neutral::destroy( const Eref& e, int stage )
{
	vector< Id > tree;
	unsigned int numDescendants = buildTree( e, tree );
	/*
	cout << "Neutral::destroy: id = " << e.id() << 
		", name = " << e.element()->getName() <<
		", numDescendants = " << numDescendants << endl;
		*/
	assert( numDescendants == tree.size() );
	Element::destroyElementTree( tree );
}

/**
 * Request conversion of data into a blockDataHandler subclass,
 * and to carry out node balancing of data as per args.
 */
void Neutral::blockNodeBalance( const Eref& e, 
	unsigned int, unsigned int, unsigned int )
{
}

/**
 * Request conversion of data into a generalDataHandler subclass,
 * and to carry out node balancing of data as per args.
 */
void Neutral::generalNodeBalance( const Eref& e, 
	unsigned int myNode, vector< unsigned int > nodeAssignment )
{
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
		assert( mid != Msg::bad );
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

// Static function.
ObjId Neutral::parent( const Eref& e )
{
	return Neutral::parent( e.objId() );
}

ObjId Neutral::parent( ObjId oid )
{
	static const Finfo* pf = neutralCinfo->findFinfo( "parentMsg" );
	static const DestFinfo* pf2 = dynamic_cast< const DestFinfo* >( pf );
	static const FuncId pafid = pf2->getFid();

	if ( oid.id == Id() ) {
		cout << "Warning: Neutral::parent: tried to take parent of root\n";
		return Id();
	}

	MsgId mid = oid.element()->findCaller( pafid );
	assert( mid != Msg::bad );

	ObjId pa = Msg::getMsg( mid )->findOtherEnd( oid );
	return pa;
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
		if ( mid == Msg::bad ) {
			cout << "Error: Neutral::path:Cannot follow msg of ObjId: " <<
				   	e.objId() << " for func: " << pafid << endl;
			break;
		}
		curr = Msg::getMsg( mid )->findOtherEnd( curr );
		pathVec.push_back( curr );
	}
	if ( pathVec.size() <= 1 )
		return "/";
	for ( unsigned int i = 1; i < pathVec.size(); ++i ) {
		ss << "/";
		ObjId& oid = pathVec[ pathVec.size() - i - 1 ];
		ss << oid.element()->getName();
		if ( oid.element()->numData() > 1 )
			ss << "[" << oid.dataId << "]";
		if ( oid.element()->numField( oid.dataId ) > 1 )
			ss << "[" << oid.fieldIndex << "]";
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
	if ( field.length() < 8 )
		return 0;
	if ( field.substr( 0, 4 ) == "set_" ) {
		if ( field == "set_name" )
			return 1;
		if ( field == "set_group" )
			return 1;
		if ( field == "set_lastDimension" ) // This is the critical one!
			return 1;
	}
	return 0;
}

