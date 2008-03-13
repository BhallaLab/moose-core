/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SLOT_H
#define _SLOT_H

/**
 * The 'Slot' class identifies the MsgSrc and Function in the 
 * 'send' call to send messages out. 
 * Slots are typically created statically on startup and used
 * internally by Element functions whenever they need to send messages.
 */
class Slot
{
	public:
		/*
		Slot( unsigned short msg, unsigned short func )
			: msg_( msg ), func_( func )
		{;}
		*/

		Slot()
			: msg_( 0 ), func_( 0 )
		{;}
		Slot( unsigned int msg, unsigned int func )
			: msg_( msg ), func_( func )
		{;}

		const unsigned int msg() const {
			return msg_;
		}

		const unsigned int func() const {
			return func_;
		}

		bool operator==( const Slot& other ) {
			return ( msg_ == other.msg_ && func_ == other.func_ );
		}

	private:
		unsigned int msg_;
		unsigned int func_;
};

#endif // _SLOT_H
