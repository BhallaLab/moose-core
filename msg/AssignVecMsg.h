/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ASSIGN_VEC_MSG_H
#define _ASSIGN_VEC_MSG_H

/**
 * This is a specialized msg type used during field set and gets. It is
 * always anchored at e1 to the Shell, and at e2 to the actual target.
 * It always has the same MsgId of one.
 */

class AssignVecMsg: public Msg
{
	friend void Msg::initMsgManagers(); // for initializing Id.
	public:
		AssignVecMsg( MsgId mid, Eref e1, Element* e2 );
		~AssignVecMsg();

		void exec( const char* arg, const ProcInfo* p) const;

		Eref firstTgt( const Eref& src ) const;

		Id managerId() const;

		ObjId findOtherEnd( ObjId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;

		void addToQ( const Element* src, Qinfo& q, const ProcInfo* p,
			MsgFuncBinding i, const char* arg ) const;

		unsigned int srcToDestPairs(
			vector< DataId >& src, vector< DataId >& dest) const;

		/// Return the first DataId
		DataId getI1() const;

		/// Setup function for Element-style access to Msg fields.
		static const Cinfo* initCinfo();
	private:
		DataId i1_;

		static Id managerId_;
};

#endif // _ASSIGN_VEC_MSG_H
