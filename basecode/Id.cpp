/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "IdManager.h"
#include "Fid.h"

#include "../shell/Shell.h"

const unsigned int Id::BadIndex = UINT_MAX;
const unsigned int Id::AnyIndex = UINT_MAX - 1;
const unsigned int BAD_ID = ~0;
const unsigned int MAX_ID = 1000000;

//////////////////////////////////////////////////////////////
//	Id creation
//////////////////////////////////////////////////////////////

Id::Id()
	: id_( 0 ), index_( 0 )
{;}

Id::Id( unsigned int i )
	: id_( i ), index_( 0 )
{;}

///\todo Lots of stuff to do here, mostly to refer to Shell's operations.
Id::Id( const string& path, const string& separator )
{
	*this = Shell::path2eid( path, separator );
}

// static func
Id Id::childId( Id parent )
{
	return Id( manager().childId( parent.id_ ) );
}

// static func
Id Id::scratchId()
{
	return Id( manager().scratchId() );
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

/**
 * Static func to extract an id from a string. We need to accept ids
 * out of the existing range, but it would be nice to have a heuristic
 * on how outrageous the incoming value is.
 */
Id Id::str2Id( const std::string& s )
{
	unsigned int val = atoi( s.c_str() );
	return Id( val );
}

Id Id::assignIndex( unsigned int index ) const
{
	Id i( id_ );
	i.index_ = index;
	return i;
}

/**
 *
 * Deprecated
void Id::setIndex(unsigned int index){
	id_ = index;
}
*/

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
	char temp[40];
	if ( id.index_ == 0 )
		sprintf( temp, "%d", id.id_ );
	else
		sprintf( temp, "%d[%d]", id.id_, id.index_ );
	return temp;
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

bool Id::isScratch() const
{
	return manager().isScratch( id_ );
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


void Id::setNodes(  unsigned int myNode, unsigned int numNodes,
			vector< Element* >& post )
{
	manager().setNodes( myNode, numNodes, post );
}
