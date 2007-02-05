/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ELEMENT_H
#define _ELEMENT_H

/**
 * The Element class handles all the MOOSE infrastructure: messages,
 * field information and class information. It manages the actual
 * data class through a generic char* pointer.
 * Here we start out with a generic base class Element, which will
 * be specialized as needed for arrays and other things.
 */

class Element
{
	public:
		Element();

		virtual ~Element();

		/// Returns the name of the element
		virtual const std::string& name( ) const = 0;

		/// Sets the name of the element.
		virtual void setName( const std::string& name ) = 0;

		/// Looks up the specific indexed conn
		virtual vector< Conn >::const_iterator 
				lookupConn( unsigned int i ) const = 0;

		/// Looks up the specific indexed conn, allows modification.
		virtual vector< Conn >::iterator 
				lookupVariableConn( unsigned int i ) = 0;

		/// Finds the index of the specified conn
		virtual unsigned int 
				connIndex( const Conn* ) const = 0;

		/// Returns the size of the conn vector.
		virtual unsigned int connSize() const = 0;

		/// Sets up a connection between previously created Conn entries
		virtual void connect( unsigned int myConn,
			Element* targetElement, unsigned int targetConn) = 0;

		/// Deletes a connection, one side has enough info.
		virtual void disconnect( unsigned int connIndex ) = 0;

		/// Deletes the local half of the connection.
		virtual void deleteHalfConn( unsigned int connIndex ) = 0;

		/// True if element is marked for deletion.
		virtual bool isMarkedForDeletion() const = 0;

		/// Before actual delete, mark all victims for message cleanup.
		virtual void prepareForDeletion( bool stage ) = 0;

		/** Makes a source conn and returns the index of the new conn.
		 * It is placed on the appropriate location
		 * according to which MsgSrc and MsgDests (if any) are involved
		 * in this message. The FuncList of recvFuncs is compared
		 * with existing ones to see if the new conn can just be
		 * appended on an existing MsgSrc, or if a new one needs to
		 * be set up to handle it.
		 * The MsgDests have to be included here
		 * because the conn may be for a shared message.
		 * All the Srcs and Dests are updated in case the new Conn
		 * has altered indices.
		 */
		virtual unsigned int insertConnOnSrc(
				unsigned int src, FuncList& rf,
				unsigned int dest, unsigned int nDest
		) = 0;

		/** Makes a dest conn and returns the index of the new conn.
		 * Here we only have to worry about placing it on one or more
		 * dests. If it were a shared message with any srcs then the
		 * insertConnOnSrc function would have applied.
		 * All the Dests are updated in case the new Conn
		 * has altered indices. The Srcs are safe in the lower indices.
		 */
		virtual unsigned int insertConnOnDest(
				unsigned int dest, unsigned int nDest
		) = 0;

		/**
		 * Returns a pointer to the data stored on this Element.
		 */
		virtual void* data() const = 0;

		/** Returns a Finfo that matches the path given by 'name'.
		 * This can be a general path including field indirection
		 * and indices. If necessary the function will generate
		 * a dynamic Finfo to handle the request. For this reason
		 * it cannot be a const function of the Element.
		 */
		virtual const Finfo* findFinfo( const string& name ) = 0;

		/**
		 * Returns finfo ptr associated with specified conn index.
		 * For ordinary finfos, this is a messy matter of comparing
		 * the conn index with the ranges of MsgSrc or MsgDest
		 * entries associated with the finfo. However, this search
		 * happens after the dynamic finfos on the local element.
		 * For Dynamic Finfos, this is fast: it just scans through
		 * all finfos on the local finfo_ vector to find the one that 
		 * has the matching connIndex.
		 */
		virtual const Finfo* findFinfo( unsigned int connIndex )
				const = 0;

		/**
		 * Checks that specified finfo does in fact come from this
		 * Element. Used for paranoia checks in some functions,
		 * though perhaps this can later be phased out by using
		 * encapsulation of finfos and elements into a Field object.
		 */
		virtual unsigned int listFinfos(
			vector<	const Finfo* >& flist ) const = 0;

		/**
		 * Appends a new Finfo onto the Element. Typically this new
		 * Finfo is a Dynamic Finfo used to store messages that are
		 * not precompiled, and therefore need to be allocated on the
		 * fly. It is also used once at startup to insert the 
		 * main Finfo for the object, typically ThisFinfo, that holds
		 * the class information.
		 */
		virtual void addFinfo( Finfo* f ) = 0;

		/**
		 * Returns true if the specified connection is on the specified
		 * MsgSrc
		 */
		virtual bool isConnOnSrc(
			unsigned int src, unsigned int conn ) const = 0;

		/**
		 * Returns true if the specified connection is on the specified
		 * MsgDest
		 */
		virtual bool isConnOnDest(
			unsigned int dest, unsigned int conn ) const = 0;

		/**
		 * Returns the root element. This function is declared
		 * in Neutral.cpp, because that is the data type of the 
		 * root Element.
		 */
		static Element* root();

		unsigned int id() const {
			return id_;
		}

		static Element* element( unsigned int id ) {
			if ( id < elementList.size() )
				return elementList[ id ];
			return 0;
		}

	private:
		static vector< Element* > elementList;
		unsigned int id_;
};

#endif // _ELEMENT_H
