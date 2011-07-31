/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _REDUCE_MSG_H
#define _REDUCE_MSG_H

/**
 * This is a specialized msg type used to carry out 'reduce' operations
 * where a data value is condensed from all individual object entries.
 * The reduction happens across threads and nodes. 
 */

class ReduceMsg: public Msg
{
	friend void Msg::initMsgManagers(); // for initializing Id.
	public:
		ReduceMsg( MsgId mid, Eref e1, Element* e2, 
			const ReduceFinfoBase* rfb );
		~ReduceMsg();

		void exec( const Qinfo* q, const double* arg, FuncId fid ) const;

		Eref firstTgt( const Eref& src ) const;

		Id managerId() const;

		ObjId findOtherEnd( ObjId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;

		/*
		void addToQ( const Element* src, Qinfo& q, const ProcInfo* p,
			MsgFuncBinding i, const char* arg ) const;
			*/

		unsigned int srcToDestPairs(
			vector< DataId >& src, vector< DataId >& dest) const;

		/// Return the first DataId
		DataId getI1() const;

		/// Setup function for Element-style access to Msg fields.
		static const Cinfo* initCinfo();
	private:
		DataId i1_;

		/**
		 * Pointer to source ReduceFinfoBase field on source Element.
		 * This keeps track of typing and creation of the correct
		 * Reduce subclass object.
		 */
		const ReduceFinfoBase* rfb_;

		static Id managerId_;
};

#endif // _REDUCE_MSG_H
