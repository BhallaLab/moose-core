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
	friend void initMsgManagers(); // for initializing Id.
	public:
		AssignVecMsg( Eref e1, Element* e2, MsgId mid );
		~AssignVecMsg();

		void exec( const char* arg, const ProcInfo* p) const;

		Id id() const;

		FullId findOtherEnd( FullId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;
	private:
		DataId i1_;

		static Id id_;
};

#endif // _ASSIGN_VEC_MSG_H
