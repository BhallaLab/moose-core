
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

/**
 * This class manages id lookups for elements. Ids provide a uniform
 * handle for every object, independent of which node they are located on.
 */
class Id
{
	public:
		//////////////////////////////////////////////////////////////
		//	Id creation
		//////////////////////////////////////////////////////////////
		/**
		 * Returns the root Id
		 */
		Id();

		/**
		 * Returns an id found by traversing the specified path
		 * May go off-node to find it.
		 */
		Id( const std::string& path, const std::string& separator = "/" );

		/**
		 * Destroys an Id. Doesn't do anything much.
		 */
		~Id(){}

		//////////////////////////////////////////////////////////////
		//	Element creation and deletion.
		//////////////////////////////////////////////////////////////

		static Id create( Element* e );

		Id destroy();
    
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

		unsigned int index() const;

		/**
		 * Returns the Eref to the element plus index
		 */
		Eref eref() const;

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

	private:
		Id( unsigned int id, unsigned int index );
		// static void setManager( Manager* m );
		unsigned int id_; // Unique identifier for Element*
		unsigned int index_; // Index of array entry within element.
		static vector< Element* >& elements();
};

#endif // _ID_H
