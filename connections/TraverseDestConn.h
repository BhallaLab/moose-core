/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TRAVERSE_DEST_CONN_H
#define _TRAVERSE_DEST_CONN_H

/**
 * TraverseDestConn iterates through all the inputs to a MsgDest. It 
 * represents a set of 'edges' where the Elements are nodes.
 *
 * It is the main interface to message lists and is designed to give
 * access to all the fields needed to traverse the messages.
 */
class TraverseDestConn: public Conn
{
	public:
		TraverseDestConn( 
			const vector< ConnTainer* >* ct, Eref e );

		~TraverseDestConn();

		Eref target() const {
			return c_->target();
		}
		unsigned int targetIndex() const {
			return c_->targetIndex();
		}
		int targetMsg() const {
			return c_->targetMsg();
		}
		Eref source() const {
			return c_->source();
		}
		unsigned int sourceIndex() const {
			return c_->sourceIndex();
		}
		int sourceMsg() const {
			return c_->sourceMsg();
		}
		void* data() const {
			return c_->data();
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets. Since we have a single entry in the TraverseMsgConn, all
		 * this has to do is to invalidate further good() calls.
		 */
		void increment();
		void nextElement();
		bool good() const;

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const;

		const ConnTainer* connTainer() const;

		bool isDest() const {
			return 1;
		}

	private:
		const vector< ConnTainer* >* ct_;
		Conn* c_;
		vector< ConnTainer* >::const_iterator cti_;
		Eref e_;
};

#endif // _TRAVERSE_DEST_CONN_H
