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

		/// Returns the name of the element class
		virtual const std::string& className( ) const = 0;

		/// Looks up the specific indexed conn
		virtual vector< Conn >::const_iterator 
				lookupConn( unsigned int i ) const = 0;

		/// Looks up the specific indexed conn, allows modification.
		virtual vector< Conn >::iterator 
				lookupVariableConn( unsigned int i ) = 0;

		/// Finds the index of the specified conn
		virtual unsigned int 
				connIndex( const Conn* ) const = 0;

		/**
		 * Finds the relative index of a conn arriving at this element.
		 * Relative index means it is based on the specified
		 * msgDest slot, so that the first conn in this slot would
		 * have an index of 0.
		 */
		virtual unsigned int connDestRelativeIndex(
				const Conn& c, unsigned int slot ) const = 0;
		
		/// Returns the size of the conn vector.
		virtual unsigned int connSize() const = 0;

		/**
		 * This function returns the iterator to conn_ at the beginning
		 * of the Src range specified by i. Note that we don't need
		 * to know how the Element handles MsgSrcs here.
		 */
		virtual vector< Conn >::const_iterator
				connSrcBegin( unsigned int src ) const = 0;

		/**
		 * This function returns the iterator to conn_ at the end
		 * of the Src range specified by the src arg.
		 * End here is in the same
		 * sense as the end() operator on vectors: one past the last
		 * entry. Note that we don't need
		 * to know how the Element handles MsgSrcs here.
		 * Note also that this call does NOT follow the linked list of
		 * Srcs to the very end. It applies only to the Conns that are
		 * on the src_ entry given by the src argument.
		 * If you want to follow the linked list, use the nextSrc
		 * function.
		 * If you want to go to the very end of the linked list, use
		 * connSrcVeryEnd
		 */
		virtual vector< Conn >::const_iterator
				connSrcEnd( unsigned int src ) const = 0;

		/**
		 * This function returns to the iterator to conn_ at the end
		 * of the linked list of srcs starting with the src arg.
		 * End here is in the same
		 * sense as the end() operator on vectors: one past the last.
		 */
		virtual vector< Conn >::const_iterator
				connSrcVeryEnd( unsigned int src ) const = 0;

		/**
		 * Returns the index of the next src entry on this
		 * linked list of srcs.
		 * Returns zero at the end of the list.
		 */
		virtual unsigned int nextSrc( unsigned int src ) const = 0;

		/**
		 * This function returns the iterator to conn_ at the beginning
		 * of the Dest range specified by i.
		 */
		virtual vector< Conn >::const_iterator
				connDestBegin( unsigned int dest ) const = 0;

		/**
		 * This function returns the iterator to conn_ at the end
		 * of the Dest range specified by i. End here is in the same
		 * sense as the end() operator on vectors: one past the last
		 * entry.
		 */
		virtual vector< Conn >::const_iterator
				connDestEnd( unsigned int dest ) const = 0;

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
		 * Finds the local Finfos associated with this Element.
		 * Note that these are variable. Typically they are Dynamic
		 * Finfos.
		 * Returns number of Finfos found.
		 */
		virtual unsigned int listLocalFinfos( vector< Finfo* >& flist )
				= 0;

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

		static Element* element( unsigned int id );

		static Element* lastElement() {
				return elementList().back();
		}

		static unsigned int numElements();

		///////////////////////////////////////////////////////////////
		// Functions for the copy operation
		///////////////////////////////////////////////////////////////
		/**
		 * This function does a deep copy of the current element 
		 * including all messages. Returns the base of the copied tree.
		 * It attaches the copied element tree to the parent.
		 */
		virtual Element* copy( Element* parent ) const = 0;
		/**
		 * True if current element descends from the specified ancestor.
		 */
		virtual bool isDescendant( const Element* ancestor ) const = 0;

		/**
		 * This function fills up the map with current element and
		 * all its descendants. Returns the root element of the
		 * copied tree. The first entry in the map is the original
		 * The second entry in the map is the copy.
		 * The function does NOT fix up the messages.
		 */
		virtual Element* innerDeepCopy( 
				map< const Element*, Element* >& tree )
				const = 0;

		/**
		 * This function replaces Element* pointers in the conn_ vector
		 * with corresponding ones from the copied tree.
		 */
		virtual void replaceCopyPointers(
						map< const Element*, Element* >& tree ) = 0;
	protected:
		/**
		 * This function copies the element, its data and its
		 * dynamic Finfos. What it does not do is to replace
		 * any pointers to other elements in the Conn array.
		 * It does not do anything about the element hierarchy
		 * either, because that is also handled through messages,
		 * ie., the Conn array.
		 * The returned Element is dangling in memory: No parent
		 * or child.
		 */
		virtual Element* innerCopy() const = 0;


	private:
		static vector< Element* >& elementList();
		// static vector< Element* > elementList;
		unsigned int id_;
};

#endif // _ELEMENT_H
