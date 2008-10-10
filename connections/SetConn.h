/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SET_CONN_H
#define _SET_CONN_H

/**
 * This Conn is used as a handle when doing assignments, ie., the set
 * command. All it really needs to know are the target Element and eIndex.
 */
class SetConn: public Conn
{
	public:
		SetConn( Element* e, unsigned int eIndex )
			: Conn( 0 ), e_( e, eIndex )
		{;}

		SetConn( Eref e )
			: Conn( 0 ), e_( e )
		{;}

		~SetConn()
		{;}

		Eref target() const {
			return e_;
		}
		unsigned int targetIndex() const {
			return 0;
		}
		int targetMsg() const {
			return 0;
		}
		Eref source() const {
			return e_;
		}
		unsigned int sourceIndex() const {
			return 0;
		}
		int sourceMsg() const {
			return 0;
		}
		void* data() const {
			return e_.e->data( e_.i );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment()
		{ ; }
		void nextElement()
		{ ; }
		bool good() const {
			return 0;
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const {
			return new SetConn( *this );
		}

		const ConnTainer* connTainer() const {
			return 0;
		}

		bool isDest() const {
			return 0;
		}

	private:
		Eref e_;
};

#endif // _SET_CONN_H
