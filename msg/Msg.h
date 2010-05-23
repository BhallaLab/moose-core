/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MSG_H
#define _MSG_H

/**
 * Manages data flow between two elements. Is always many-to-many, with
 * assorted variants.
 */

class Msg
{
	public:
		Msg( Element* e1, Element* e2, Id managerId );

		Msg( Element* e1, Element* e2, MsgId mid, Id managerId );
		virtual ~Msg();

		/**
		 * Deletes a message identified by its mid.
		 */
		static void deleteMsg( MsgId mid );

		/**
 		* Initialize the Null location in the Msg vector.
 		*/
		static void initNull();

		/**
		 * Calls Process on e1.
		 */
		virtual void process( const ProcInfo *p ) const;

		/**
		 * Report if the msg accepts input from the DataId specified
		 * in the Qinfo. Note that the direction of the message is also
		 * important to know in order to figure this out.
		 * Usually the answer is yes, regardless of DataId.
		 */
		virtual bool isMsgHere( const Qinfo& q ) const {
			return 1;
		}

		
		/**
		 * Execute func( arg ) on all relevant indices of target
		 */
		virtual void exec( const char* arg, const ProcInfo* p ) const = 0;

		/**
		 * Return the Id of the managing Element. Each derived Msg class
		 * has its own manager. This is predefined at initialization.
		 */
		virtual Id id() const = 0;

		/**
		 * Return the Eref of the managing object.
		 */
		virtual Eref manager( Id id ) const;

		/**
		 * Fills in the specified dataId for this msg.
		 */
		void setDataId( unsigned int di ) const;

		/**
		 * Return the first element
		 */
		Element* e1() const {
			return e1_;
		}

		/**
		 * Return the second element
		 */
		Element* e2() const {
			return e2_;
		}

		/**
		 * return the Msg Id.
		 */
		MsgId mid() const {
			return mid_;
		}

		/**
		 * Find the other end of this Msg. In most cases this is a 
		 * straightforward return of e1 or e2, plus perhaps a DataId.
		 * But in some complex msgs we need to figure out
		 * DataIds that match with the target.
		 * In many-to-one cases we just return the first entry.
		 * If no Element match, return FullId( Id(), DataId::bad() )
		 * If Element e matches but not DataId, return 
		 * FullId( e.id(), DataId::bad() )
		 */
		virtual FullId findOtherEnd( FullId ) const = 0;

		/**
		 * Looks up the message on the global vector of Msgs. No checks,
		 * except assertions in debug mode.
		 */
		static const Msg* getMsg( MsgId m );

		/**
		 * Returns the Msg if the MsgId is valid, otherwise returns 0.
		 * Utility function to check if Msg is OK. Used by diagnostic
		 * function Qinfo::reportQ. Normal code uses getMsg above.
		 */
		static Msg* safeGetMsg( MsgId m );

		/**
		 * The zero MsgId, used as the error value.
		 */
		static const MsgId badMsg;

		/**
		 * A special MsgId used for Shell to get/set values
		 */
		static const MsgId setMsg;

	protected:
		Element* e1_;
		Element* e2_;
		MsgId mid_; // Index of this msg on the msg_ vector.

		/// Manages all Msgs in the system.
		static vector< Msg* > msg_;

		/**
		 * Manages garbage collection for deleted messages. Deleted
		 * MsgIds are kept here so that they can be reused.
		 */
		static vector< MsgId > garbageMsg_;

		/**
		 * Manages lookup of DataIds from MsgIds. This is the index of
		 * the MsgManager data entry in the appropriate Element, for the
		 * specified MsgId.
		 * Used to find and set
		 * msg fields. We only need the data part of the DataId, so this
		 * is just an unsigned int.
		 */
		static vector< unsigned int > lookupDataId_;
};

#endif // _MSG_H
