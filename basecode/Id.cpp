/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "IdManager.h"
#include "Fid.h"

#include "../shell/Shell.h"
#include "../parallel/ProxyElement.h"

const unsigned int Id::BadIndex = UINT_MAX;
const unsigned int Id::AnyIndex = UINT_MAX - 1;

const unsigned int BAD_ID = UINT_MAX;
// const unsigned int MAX_ID = 1000000;

const unsigned int Id::BadNode = UINT_MAX;
const unsigned int Id::UnknownNode = UINT_MAX - 1;
const unsigned int Id::GlobalNode = UINT_MAX - 2;

//////////////////////////////////////////////////////////////
//	Id creation
//////////////////////////////////////////////////////////////

Id::Id()
	: id_( 0 ), index_( 0 )
{;}

Id::Id( Nid nid )
	: id_( nid.id() ), index_( nid.index() )
{ 
	if ( id_ != BAD_ID )
		this->setNode( nid.node() );
}

Id::Id( unsigned int i, unsigned int index )
	: id_( i ), index_( index )
{;}

Id::Id( const string& path, const string& separator )
{
	*this = Shell::path2eid( path, separator, 0 ); // flag says parallel
}

Id Id::localId( const string& path, const string& separator )
{
	return Shell::path2eid( path, separator, 1 ); // flag says local only
}

// static func
void Id::dumpState( ostream& stream )
{
	manager().dumpState( stream );
}

// static func
unsigned int Id::childNode( Id parent )
{
	return manager().childNode( parent.id_ );
}

// static func
Id Id::childId( Id parent )
{
	return Id( manager().childId( parent.id_ ) );
}

// static func
Id Id::newId()
{
	return Id( manager().newId() );
}

// static func
Id Id::initId()
{
	return Id( manager().initId() );
}

// static func
Id Id::makeIdOnNode( unsigned int node )
{
	return Id( manager().makeIdOnNode( node ) );
}

// static func
Id Id::shellId()
{
	return Id( 1 );
}

// static func
Id Id::postId( unsigned int node )
{
	return Id( 2, node );
}

/**
 * Static func to extract an id from a string. We need to accept ids
 * out of the existing range, but it would be nice to have a heuristic
 * on how outrageous the incoming value is.
 */
Id Id::str2Id( const std::string& s )
{
	// unsigned int val = atoi( s.c_str() );
	return Id( s );
}

Id Id::assignIndex( unsigned int index ) const
{
	Id i( id_ );
	i.index_ = index;
	return i;
}

unsigned int Id::newIdBlock( unsigned int size, unsigned int node )
{
	return manager().newIdBlock( size, node );
}

IdGenerator Id::generator( unsigned int node )
{
	return manager().generator( node );
}

//////////////////////////////////////////////////////////////
//	Id manager static access function. Private.
//////////////////////////////////////////////////////////////

IdManager& Id::manager()
{
	static IdManager manager;
	return manager;
}

//////////////////////////////////////////////////////////////
//	Id info
//////////////////////////////////////////////////////////////

// static func to convert id into a string.
string Id::id2str( Id id )
{
	/*
	char temp[40];
	if ( id.index_ == 0 )
		sprintf( temp, "%d", id.id_ );
	else
		sprintf( temp, "%d[%d]", id.id_, id.index_ );
	return temp;
	*/
	return id.path();
}

// Function to convert it into its fully separated path.
string Id::path( const string& separator) const 
{
	return Shell::eid2path( *this );
}

/**
 * Here we work with a single big array of all ids. Off-node elements
 * are represented by their postmasters. When we hit a postmaster we
 * put the id into a special field on it. Note that this is horrendously
 * thread-unsafe.
 * \todo: I need to replace the off-node case with a wrapper Element
 * return. The object stored here will continue to be the postmaster,
 * and when this is detected it will put the postmaster ptr and the id
 * into the wrapper element. The wrapper's own id will be zero so it
 * can be safely deleted.
 */
Element* Id::operator()() const
{
	return manager().getElement( *this );
}

Eref Id::eref() const 
{
	return Eref( manager().getElement( *this ), index_ );
}

unsigned int Id::node() const 
{
	return manager().findNode( id_ );
}

bool Id::isGlobal() const 
{
	return manager().isGlobal( id_ );
}

void Id::setGlobal()
{
	manager().setGlobal( id_ );
}

void Id::setNode( unsigned int node )
{
	manager().setNode( id_, node );
}

Id Id::lastId()
{
	return manager().lastId();
}

Id Id::badId()
{
	static Id b( BAD_ID );

	return b;
}

//////////////////////////////////////////////////////////////
//	Id status
//////////////////////////////////////////////////////////////

bool Id::bad() const
{
	return id_ == BAD_ID;
}

bool Id::good() const
{
	return ( !( bad() || outOfRange() || zero() ) );
}

bool Id::zero() const
{
	return id_ == 0;
}

bool Id::outOfRange() const
{
	return manager().outOfRange( id_ );
}

bool Id::isProxy() const
{
#ifdef USE_MPI
	if ( good() ) {
		Element* e = manager().getElement( *this );
		if ( e ) {
			if ( dynamic_cast< ProxyElement* >( e ) ) {
				return 1;
			} else {
				cout << "Error: Id::isProxy(): Found a regular element when looking for a proxy\n";
				assert( 0 ); // Should never look for a proxy and find
				// a regular element.
			}
		}
	}
	return 0;
#else // USE_MPI
	return 0;
#endif // USE_MPI
}

//////////////////////////////////////////////////////////////
//	Id utility
//////////////////////////////////////////////////////////////

ostream& operator <<( ostream& s, const Id& i )
{
	if ( i.index_ == 0 )
		s << i.id_;
	else 
		s << i.id_ << "[" << i.index_ << "]";
	return s;
}

istream& operator >>( istream& s, Id& i )
{
	s >> i.id_;
	return s;
}
//////////////////////////////////////////////////////////////
//	Id assignment
//////////////////////////////////////////////////////////////

// a private function
bool Id::setElement( Element* e )
{
	return manager().setElement( id_, e );
}

/**
 * Deprecated
void Id::setNodes(  unsigned int myNode, unsigned int numNodes )
{
	manager().setNodes( myNode, numNodes );
}
*/

//////////////////////////////////////////////////////////////
//	Nid stuff
//////////////////////////////////////////////////////////////
Nid::Nid()
	: Id(), node_( 0 )
{;}

Nid::Nid( Id id )
	: Id( id ), node_( id.node() )
{;}

Nid::Nid( Id id, unsigned int node ) 
	: Id( id ), node_( node )
{;}
