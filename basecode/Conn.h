/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

typedef unsigned int ConnId;

/**
 * Mid-level message handler. Manages multiple messages, and appropriate
 * funcs.
 */
class Conn
{
	public:
		~Conn();
		/**
		 * asend goes through all child Msgs with the specified FuncId
		 * and adds them into the Element Queue.
		 */
		void asend( const Element* e, Qinfo& q, const char* arg ) const;

		/**
		 * Goes through all Msgs, deallocating them. This is separate
		 * from the Conn destructor, to allow us to pass Conns around
		 * without invoking destructors.
		 */
		void clearConn();

		/**
		 * Tsend goes through child Msgs looking for an Element matching
		 * the target, and calls the matching Id.
		 */
		void tsend( const Element* e, Id target, Qinfo& q, 
			const char* arg ) const;

		void tsend( const Element* e, unsigned int targetIndex, Qinfo& q, 
			const char* arg ) const;

		/**
		 * ClearQ calls clearQ on all Msgs.
		 */
		void clearQ();

		/**
		 * Add a msg to the list
		 */
		void add( Msg* m );

		/**
		 * Drop a msg from the list
		 */
		void drop( Msg* m );
	private:
		vector< Msg* > m_;
};

