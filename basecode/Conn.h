/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CONN_H
#define _CONN_H

/**
 * This definition is used to indicate that a conn is a dummy one.
 */
#define MAXUINT (unsigned int)(~0)

/**
 * This class handles connections. Connections are the underlying
 * linkages upon which messages run: they are like wires, bidirectional.
 * Mutiple messages can run on any given wire.
 * An array of Conns is present in each SimpleElement, 
 * to enable bidirectional traversal of messages.
 *
 * The Conn structure has two design requirements: 
 * First, it must provide complete traversability in either direction.
 * This is achieved because we can identify the remote Conn with the
 * combination of Element* and target index. With a little more work
 * involving lookup of matching indices on the Element, we can also
 * identify the remote MsgDest or MsgSrc.
 * Second, it must provide complete information for the RecvFunc
 * that handles message operations on the destination object. 
 * Most RecvFuncs operate only on the Element*, but some need to know
 * their own index (which is the target index). Some need to go back
 * to the originating object, which involves looking up the matching
 * Conn on the target Element.
 *
 * \todo We need to provide a range of functions to handle changing
 * of target Elements, or reindexing following changes in the target
 * array of Conns.
 */
class Conn
{
		public:
			Conn()
					: e_( 0 ), index_( 0 )
			{;}

			Conn( Element* e, unsigned int targetConnIndex )
					: e_( e ), index_( targetConnIndex )
			{;}

			
			/**
			 * Returns the originating Element for this Conn
			 * Used infrequently, involves multiple lookups.
			 */
			Element* sourceElement() const;

			/**
			 * Returns the target Element for this Conn. Fast.
			 * Used in every message call, just returns local field.
			 */
			Element* targetElement() const {
					return e_;
			}

			/**
			 * Returns the index of this Conn on the originating
			 * Element.
			 * Used infrequently, involves multiple lookups.
			 * Use this call only if you have already got a handle
			 * on the originating Element.
			 */
			unsigned int sourceIndex( const Element* e ) const;
			
			/**
			 * Returns the index of this Conn on the originating
			 * Element, using internal information only.
			 * Used infrequently, involves multiple lookups.
			 * Use this call for preference as it is safer.
			 */
			unsigned int sourceIndex( ) const;

			/**
			 * Returns the index of the target Conn. Fast.
			 * Used in every message call, just returns local field
			 */
			unsigned int targetIndex() const {
					return index_;
			}

			/**
			 * This function tells the target conn that the 
			 * index of the source has changed to j.
			 */
			void updateIndex( unsigned int j );
			
			void set( Element* e, unsigned int index );

			/**
			 * This utility function gets the data pointer from the
			 * targetElement. It is used very frequently in recvFuncs.
			 */
			void* data() const;

			/**
			 * Utility function used only by Copy.cpp as far as I know
			 */
			void replaceElement( Element* e ) {
				e_ = e;
			}

		private:
			
			/// e_ points to the target element.
			Element* e_;


			/** index_ is the absolute index of the target conn, in
			 * the conn_ array on the target element.
			 */
			unsigned int index_;	
};

#endif // _CONN_H
