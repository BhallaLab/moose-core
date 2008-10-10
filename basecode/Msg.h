/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MSG_H
#define _MSG_H

/**
 * The MsgSrc class specifies the ranges of conns to be used for
 * messaging, and specifies which functions to be used within the
 * range. In case there are additional ranges with different funcs
 * but the same src, we use the next_ index to point to a location
 * further up in the array, past the predefined src range.
 * Used for the source of simple messages, and for either end of 
 * bidirectional shared messages.
 */
class Msg
{
	public:
		Msg();
		/**
		 * Destructor needs to clean up all connections
		 */
		~Msg();

		/**
		 * Begin const_iterator for the ConnTainers on this Msg.
		 * Used in most iterations through the ConnTainers.
		 */
		vector< ConnTainer* >::const_iterator begin() const {
			return c_.begin();
		}
		/**
		 * End const_iterator for the ConnTainers on this Msg.
		 * Used in most iterations through the ConnTainers.
		 */
		vector< ConnTainer* >::const_iterator end() const {
			return c_.end();
		}

		/**
		 * Begin iterator for the ConnTainers on this Msg. This
		 * is used for those rare cases where the user wants to mess
		 * with the ConnTainers.
		 */
		vector< ConnTainer* >::iterator varBegin() {
			return c_.begin();
		}
		/**
		 * End iterator for the ConnTainers on this Msg. This
		 * is used for those rare cases where the user wants to mess
		 * with the ConnTainers.
		 */
		vector< ConnTainer* >::iterator varEnd() {
			return c_.end();
		}

		/**
		 * Size of the ConnTainer vector
		 * \todo: Should we include the 'next' size here too?
		 */
		unsigned int size() const;

		/**
		 * Iterator to the conn selected by the 'tgt' index.
		 */
		Conn* findConn( Eref e, unsigned int tgt, unsigned int funcId ) const;

		/**
		 * Follows through the link list of msgs to find one that matches
		 * the funcId. 
		 * Used by findExistingConnTainer
		 */
		Msg* matchByFuncId( Element* e, unsigned int funcId );

		/**
 		* Adds a new message either by finding an existing ConnTainer that
 		* matches, and inserting the eIndices in that, or by creating a new 
 		* Conntainer.
		*
		* This is meant to be used internally by Finfos to set up messages.
 		*
 		* Returns true on success.
 		*/
		
		static bool add( Eref src, Eref dest,
			int srcMsg, int destMsg,
			unsigned int srcIndex, unsigned int destIndex,
			unsigned int srcFuncId, unsigned int destFuncId,
			unsigned int connTainerOption );

		/**
		 * Add a new message using the specified ConnTainer.
		 * The e1 (source ) and e2 (dest), are in the ConnTainer, as are
		 * m1 and m2 which indicate source and dest msg indices.
		 * The funcId1 is the source funcId, which is going to be used
		 * at the dest, but is optional so it may be zero.
		 * The funcId2 is the dest funcId, which must be nonzero and will
		 * be used when the source calls the dest.
		 * Later I may relax the directional restrictions.
		 *
		 * Returns true on success.
		 */
		static bool add( ConnTainer* ct,
			unsigned int funcId1, unsigned int funcId2 );

		/**
		 * This variant of drop initiates the removal of a specific 
		 * local ConnTainer, identified by index as doomed.
		 */
		bool drop( Element* e, unsigned int doomed );

		/**
		 * This variant of drop initiates the removal of a specific 
		 * local ConnTainer
		 */
		bool drop( Element* e, const ConnTainer* doomed );

		/**
		 * Drops all messages emanating from this Msg. Often used in
		 * rescheduling.
		 */
		void dropAll( Element* e );

		/**
		 * innerDrop eliminates an identified ConnTainer.
		 * This is NOT the call to initiate removing a connection.
		 * It is called on the other end of the message from the one
		 * set up for deletion, and assumes that the rest of the message
		 * will be taken care of by the initiating function.
		 * Assumes that the element checks first to see if it is also
		 * doomed. If so, it should not bother with this call.
		 * If the element is to survive, only then it goes through the
		 * bother of erasing.
		 *
		 * Note that this does not do garbage collection if a 'next' Msg is 
		 * emptied. Something to think about, much later.
		 */
		bool innerDrop( const ConnTainer* doomed );

		/**
		 * Utility function for dropping target, whether it is 
		 * on a pure dest or another Msg.
		 * Does not clear the ConnTainer.
		 * NOT the primary call to drop a message. You probably
		 * want drop( unsigned int ) and dropAll().
		 */
		static bool innerDrop( Element* remoteElm, int remoteMsgNum,
			const ConnTainer* d );

		/**
		 * Drops all external messages emanating from this Msg, that is
		 * messages to Elements not MarkedForDeletion. Used when a
		 * tree of elements are to be deleted. After this operation the
		 * messages within the tree remain, but can simply be deallocated
		 * without careful removing of messages, as the whole tree is
		 * being deleted.
		 */
		void dropRemote();

		/**
 		* Deletes all messages originating from outside the current tree
		* onto the dest_ of an Element to be deleted. Same as innerDump,
		* the messages within the tree are unaffected.
 		* This is called from the viewpoint of the destination ConnTainer
 		* on the Element to be deleted.
 		* A static function, nothing much to do with the Msg class.
 		* Here only because it keeps all the related deletion
 		* operations in one place.
 		*/
		static void dropDestRemote( vector< ConnTainer* >& ctv  );

		/**
		 * Drops all messages during deletion. Assumes that all targets
		 * are also scheduled for deletion.
		 * So the function simply frees all outgoing ConnTainers, and 
		 * then frees the ConnTainer vector. It does not have to 
		 * clean up anything on the targets because they too are doomed.
		 *
		 * It only frees outgoing ConnTainers because they are shared
		 * with the doomed target message, and we don't want double deletes.
		 */
		void dropForDeletion();

		/**
		 * Utility function for putting a new ConnTainer onto a msg as
		 * specified by the funcId. If necessary it traverses through the
		 * 'next' message looking for a funcId match,  and allocates a
		 * new 'next' message for the FuncId if there are no matches.
		 */
		bool assignMsgByFuncId( 
			Element* e, unsigned int funcId, ConnTainer* ct );

		/**
		 * Returns the function identified by funcNum.
		 */
		RecvFunc func( unsigned int funcNum ) const {
			return fv_->func( funcNum );
		}


		/**
		 * True if this is the destination of a message.
		 * Undefined if the message is empty: Check for size first.
		 * The definition of message source and dest is done at Finfo
		 * setup time. For simple messages no problem. For Shared Finfos,
		 * the one that has the first 'source' entry is the source.
		 */
		bool isDest() const;

		/**
		 * Returns the ptr to the next Msg in the list.
		 */
		const Msg* next( const Element* e ) const;

		/**
		 * Returns the Id of the FuncVec
		 */
		unsigned int funcId() const {
			return fv_->id();
		}

		/**
		 * Counts the number of targets, including going through the
		 * 'next_' msg if any. May be much faster than listing.
		 */
		unsigned int numTargets( const Element* e ) const;

		/**
		 * Counts the number of messages that terminate on specified
		 * Element/eIndex. 
		 * Somewhat slow specially for Many2Many messages.
		 */
		unsigned int numSrc( const Element* e, unsigned int i ) const;

		/**
		 * Counts the number of messages that originate from specified
		 * Element/eIndex. 
		 * Somewhat slow specially for Many2Many messages.
		 */
		unsigned int numDest( const Element* e, unsigned int i ) const;

		/**
		 * Returns true if this Msg->next_ field is the same as msgNum, or
		 * if the same is true for the Msg pointed to by next_.
		 */
		bool linksToNum( const Element* e, unsigned int msgNum ) const;


		/**
		 * Makes a duplicate of the current message specified by c,
		 * to now go between e1 and e2.
		 */
		bool copy( const ConnTainer* c, Element* e1, Element* e2, bool isArray) const;

		/**
		 *  Returns True if tgt is a target of Element src.
		 *  Handles bidirectional messages too. Does not worry about
		 *  indices, on either src or dest
		 */
		bool isTarget( const Element* src, const Element* tgt ) const;

	private:
		/**
		 * This manages the ConnTainers.
		 * Each ConnTainer handles a virtual vector of Conns. The ConnTainer provides a Conn as an iterator.
		 */
		vector< ConnTainer* > c_; 

		/**
		 * Points to a global FuncList.
		 */
		const FuncVec *fv_; 

		/**
		 * Index to next Msg in Msg vector, somewhat like a linked list.
		 * Prefer index to ptr because Msg vector might be resized.
		 */
		unsigned int next_; 
};

#endif // _MSG_H
