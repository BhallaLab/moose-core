/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ID_H
#define _ID_H

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Eref;
class Element;
class IdGenerator;
class IdManager;
class Nid;
namespace pymoose{
	class PyMooseContext;
}

/**
 * This class manages id lookups for elements. Ids provide a uniform
 * handle for every object, independent of which node they are located on.
 */
class Id
{
	friend class pymoose::PyMooseContext; ///\todo: deprecate this friend.
    
	public:
		static void dumpState( ostream& stream );

		//////////////////////////////////////////////////////////////
		//	Id creation
		//////////////////////////////////////////////////////////////
		/**
		 * Returns the root Id
		 */
		Id();

		/**
		 * Sets up the Id with the Nid info. Would be automatic except
		 * that the node info also has to be transferred.
		 */
		Id( Nid nid );

		/**
		 * Returns an id found by traversing the specified path
		 * May go off-node to find it.
		 */
		Id( const std::string& path, const std::string& separator = "/" );

		/**
		 * Destroys an Id. Doesn't do anything much.
		 */
		~Id(){}
    
		/**
		 * Returns an Id found by traversing the specified path on the
		 * local node only
		 */
		static Id localId( const std::string& path, const std::string& separator = "/" );

		/**
		 * Creates a new childId based on location of parent node and
		 * whatever other heuristics the IdManager applies. Must be
		 * called only on the master node.
		 * If the object represented by childId is located on one of the 
		 * slave nodes, the shell must forward the Id to the affected
		 * node for action.
		 */
		static unsigned int childNode( Id parent );
		static Id childId( Id parent );

		/**
		 * Returns a scratch id: one in the scratch range of ids,
		 * used by local nodes if they do not expect it to be accessed
		 * by any other node. Not used by master node. This kind of 
		 * Id must never be saved or transmitted, because it may be
		 * reassigned to the general Id range.
		 */
		static Id scratchId() { return newId(); }

		static Id newId();

		static Id initId();

		/**
 		* This variant of childId forces creation of object on specified 
		* node, provided that we are in parallel mode. Otherwise it 
		* ignores the node specification and behavies like scratchId.
 		* Should only be called on master node.
 		*/
		static Id makeIdOnNode( unsigned int childNode );

		/**
		 * This returns Id( 1 ), which is the shell.
		 */
		static Id shellId();

		/**
		 * This returns Id( 2, node ), which is the postmaster of the
		 * specified node. 
		 * On a serial version this returns a neutral.
		 */
		static Id postId( unsigned int node );

		/**
		 * This creates a new Id with the same element id but a new index
		 */
		Id assignIndex( unsigned int index ) const;
		
		/**
		 * Deprecated
		// void setIndex( unsigned int index );
		 */

		//////////////////////////////////////////////////////////////
		//	Multi-node Id management
		//////////////////////////////////////////////////////////////
		static unsigned int newIdBlock( unsigned int size, unsigned int node );
		static IdGenerator generator( unsigned int node );

		//////////////////////////////////////////////////////////////
		//	Id info
		//////////////////////////////////////////////////////////////
		/**
		 * Returns the full pathname of the object on the id.
		 * This may go off-node to look for the object.
		 */
		std::string path( const std::string& separator = "/" ) const;


		/**
		 * Returns the Element pointed to by the id
		 * If it is off-node, returns an allocated wrapper element with 
		 * postmaster and id information. Calling func has to free it.
		 * This wrapper element may also point to UNKNOWN NODE, in which
		 * case the master node IdManager has to figure out where it
		 * belongs.
		 * Returns 0 on failure.
		 */
		Element* operator()() const;

		/**
		 * Returns the Element id
		 */
		unsigned int id() const {
			return id_;
		}
		/**
		 * Returns the Element index
		 */
		unsigned int index() const {
			return index_;
		}

		/**
		 * Returns the Eref to the element plus index
		 */
		Eref eref() const;

		/**
		 * Returns node on which id is located.
		 */
		unsigned int node() const;

		/**
		 * True if node is global
		 */
		bool isGlobal() const;

		/**
		 * Tells object it is a global. Used only by constructors.
		 */
		void setGlobal();

		/**
		 * Assignes node# to id. Used when creating proxy elements
		 * and Global objects.
		 */
		void setNode( unsigned int node );

		/**
		 * The most recently created id on this node
		 */
		static Id lastId();

		/**
		 * Returns a BAD_ID Id.
		 */
		static Id badId();

		/**
		 * Returns an id whose value is string-converted from the 
		 * specified string. 
		 */
		static Id str2Id( const std::string& s );

		/**
		 * Returns a string holding the ascii value of the id_ .
		 */
		static std::string id2str( Id id );

		//////////////////////////////////////////////////////////////
		//	Here we have a set of status check functions for ids.
		//////////////////////////////////////////////////////////////

		/**
		 * Checks if id has been given an error flag
		 */
		bool bad() const;

		/**
		 * Returns true only if id is not bad, not zero and not out of range
		 * Note that this more restrictive than !Id::bad().
		 */
		bool good() const;

		/**
		 * True if id points to zero object
		 */
		bool zero() const;

		/**
		 * True if id is not in allocated range.
		 */
		bool outOfRange() const;

		/**
		 * True if it is a proxy element. This is a minimal holder
		 * element on a remote node that does message redirection but
		 * for everything else refers back to the authoritative
		 * element on its node.
		 */
		bool isProxy() const;

		//////////////////////////////////////////////////////////////
		//	Comparisons between ids
		//////////////////////////////////////////////////////////////
		bool operator==( const Id& other ) const {
			return id_ == other.id_ && index_ == other.index_;
		}

		bool operator!=( const Id& other ) const {
			return id_ != other.id_ || index_ != other.index_;
		}

		bool operator<( const Id& other ) const {
			return ( id_ < other.id_ ) ||
				( id_ == other.id_ && index_ < other.index_ );
		}

		friend ostream& operator <<( ostream& s, const Id& i );
		friend istream& operator >>( istream& s, Id& i );

		//////////////////////////////////////////////////////////////
		//	Element assignment. Only allowed for a few functions.
		//////////////////////////////////////////////////////////////
		bool setElement( Element* e );
		static const unsigned int AnyIndex;
		static const unsigned int BadIndex;

		static const unsigned int BadNode;
		static const unsigned int UnknownNode;
		static const unsigned int GlobalNode;

		/**
		 * Assignment of id to specific index. This was originally
		 * a private function, but then there turned out to be so
		 * many 'friends' it was defeating the purpose.
		 * This is a handy function, but it is very likely to be
		 * misused through creation of unchecked and invalid Ids.
		 * The str2Id can be used for most test scenarios,
		 * and the rest of the time you should not be using this.
		 */
		Id( unsigned int i, unsigned int index = 0 );
	private:
		// static void setManager( Manager* m );
		unsigned int id_; // Unique identifier for Element*
		unsigned int index_; // Index of array entry within element.
		static IdManager& manager();
};

/**
 * Extension of Id class, used in passing ids around between nodes, so
 * that their node info is retained. See Shell::addParallelSrc
 */
class Nid: public Id
{
	public:
		Nid();

		Nid( Id id );

		Nid( Id id, unsigned int node );

		unsigned int node() const {
			return node_;
		}

		void setNode( unsigned int node ) {
			node_ = node;
		}
		
	private:
		unsigned int node_;
};

#endif // _ID_H
