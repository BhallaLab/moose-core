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

class Element;

/**
 * This class manages id lookups for elements. Ids provide a uniform
 * handle for every object, independent of which node they are located on.
 */
class Id
{
	// This access is needed so that main can assign the nodes to
	// the manager.
	friend int main( int argc, char** argv );
	friend class IdManager;

#ifdef DO_UNIT_TESTS
	friend void testShell();
#endif
	
	public:

		//////////////////////////////////////////////////////////////
		//	Id creation
		//////////////////////////////////////////////////////////////
		Id();

		/**
		 * Returns an id found by traversing the specified path
		 * May go off-node to find it.
		 */
		Id( const std::string& path, const std::string& separator = "/" );

		/**
		 * Creates a new childId based on location of parent node and
		 * whatever other heuristics the IdManager applies. Must be
		 * called only on the master node.
		 * If the object represented by childId is located on one of the 
		 * slave nodes, the shell must forward the Id to the affected
		 * node for action.
		 */
		static Id childId( Id parent );
		
		/**
		 * Returns a scratch id: one in the scratch range of ids,
		 * used by local nodes if they do not expect it to be accessed
		 * by any other node. Not used by master node. This kind of 
		 * Id must never be saved or transmitted, because it may be
		 * reassigned to the general Id range.
		 */
		static Id scratchId();

		/**
		 * This returns Id( 1 ), which is the shell.
		 */
		static Id shellId();

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
		 * Returns node on which id is located.
		 */
		unsigned int node() const;

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
		static string id2str( Id id );

		//////////////////////////////////////////////////////////////
		//	Here we have a set of status check functions for ids.
		//////////////////////////////////////////////////////////////

		/**
		 * Checks if id has been given an error flag
		 */
		bool bad() const;

		/**
		 * True if id points to zero object
		 */
		bool zero() const;

		/**
		 * True if id is not in allocated range.
		 */
		bool outOfRange() const;

		/**
		 * True if it is a scratch id.
		 */
		bool isScratch() const;

		//////////////////////////////////////////////////////////////
		//	Comparisons between ids
		//////////////////////////////////////////////////////////////
		bool operator==( const Id& other ) const {
			return id_ == other.id_;
		}

		bool operator!=( const Id& other ) const {
			return id_ != other.id_;
		}

		bool operator<( const Id& other ) const {
			return id_ < other.id_;
		}

		friend ostream& operator <<( ostream& s, const Id& i );
		friend istream& operator >>( istream& s, Id& i );

		//////////////////////////////////////////////////////////////
		//	Element assignment. Only allowed for a few functions.
		//////////////////////////////////////////////////////////////
		bool setElement( Element* e );
	private:
		// static void setManager( Manager* m );

		/**
		 * Assignment of id to specific index. Only done internally.
		 * This is a handy function, but it is very likely to be
		 * misused through creation of unchecked and invalid Ids.
		 * The str2Id can be used for most test scenarios,
		 * and the rest of the time you should not be using this.
		 */
		Id( unsigned int id );
		unsigned int id_;
		static IdManager& manager();
};

#endif // _ID_H
