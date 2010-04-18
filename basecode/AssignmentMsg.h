/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ASSIGNMENT_MSG_H
#define _ASSIGNMENT_MSG_H

/**
 * This is a specialized msg type used during field set and gets. It is
 * always anchored at e1 to the Shell, and at e2 to the actual target.
 * It always has the same MsgId of zero.
 */

class AssignmentMsg: public Msg
{
	public:
		AssignmentMsg( Eref e1, Eref e2, MsgId mid );
		~AssignmentMsg() {;}

		void exec( const char* arg, const ProcInfo* p) const;

	private:
		DataId i1_;
		DataId i2_;
};

#endif // _ASSIGNMENT_MSG_H
