/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MSG_MANAGER_H
#define _MSG_MANAGER_H

class MsgManager
{
	public:
		MsgManager();
		MsgManager( MsgId mid );

		Id getE1() const;
		Id getE2() const;

		void setMid( MsgId mid );
		MsgId getMid() const;

		FullId findOtherEnd( FullId end ) const;

		/**
		 * Register a new Msg into the appropriate Msg Manager.
		 */
		static void addMsg( MsgId mid, Id managerId );

		/**
		 * Remove a Msg from the appropriate Msg Manager.
		 */
		static void dropMsg( MsgId mid );

		static const Cinfo* initCinfo();
	private:
		MsgId mid_;
};

#endif // _MSG_MANAGER_H
