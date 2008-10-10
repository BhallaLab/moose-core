/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ALL_TO_ALL_CONN_H
#define _ALL_TO_ALL_CONN_H

class All2AllConn: public Conn
{
	public:
		All2AllConn( unsigned int funcIndex,
			All2AllConnTainer* a, unsigned int eIndex )
			: 
				Conn( funcIndex ), 
				a_( a ), 
				srcPos_( eIndex ), tgtPos_( 0 ), 
				end_( a->e2().size() )
		{;}

		~All2AllConn()
		{;}

		Element* targetElement() const {
			return a->e2();
		}
		unsigned int targetEindex() const {
			return tgtPos_;
		}
		unsigned int targetIndex() const {
			return tgtPos_ + a->i2();
		}
		int targetMsg() const {
			return a->msg2();
		}
		Element* sourceElement() const {
			return a->e1();
		}
		unsigned int sourceEindex() const {
			return srcPos_;
		}
		unsigned int sourceIndex() const {
			return srcPos_ + a->i1();
		}
		int sourceMsg() const {
			return a->msg1();
		}
		void* data() const {
			return a->e2()->data( tgtPos_ );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void operator++() {
			tgtPos_++;
		}
		bool good() const {
			return ( i < end );
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const {
			return new ReverseAll2AllConn( funcIndex, a_, tgtPos_ );
		}

	private:
		unsigned int srcPos_;
		unsigned int tgtPos_;
		unsigned int end_;
		All2AllContainer* a_;
};

class ReverseAll2AllConn: public Conn
{
	public:
		ReverseAll2AllConn( unsigned int funcIndex, 
			All2AllConnTainer* a, unsigned int eIndex )
			: 	Conn( funcIndex ),
				a_( a ), 
				srcPos_( eIndex ), tgtPos_( 0 ), 
				end_( a->e1().size() )
		{;}

		~ReverseAll2AllConn()
		{;}

		Element* targetElement() const {
			return a->e1();
		}
		unsigned int targetEindex() const {
			return tgtPos_;
		}
		unsigned int targetIndex() const {
			return tgtPos_ + a->i1();
		}
		int targetMsg() const {
			return a->msg1();
		}
		Element* sourceElement() const {
			return a->e2();
		}
		unsigned int sourceEindex() const {
			return srcPos_;
		}
		unsigned int sourceIndex() const {
			return srcPos_ + a->i2();
		}
		int sourceMsg() const {
			return a->msg2();
		}
		void* data() const {
			return a->e1()->data( tgtPos_ );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			tgtPos_++;
		}
		bool good() {
			return ( i < end );
		}
		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const {
			return new All2AllConn( funcIndex, a_, tgtPos_ );
		}

	private:
		unsigned int srcPos_;
		unsigned int tgtPos_;
		unsigned int end_;
		All2AllContainer* a_;
};

class All2AllConnTainer: public ConnTainer
{
	public:
		All2AllConnTainer( Element* e1, Element* e2,
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 )
			:
			ConnTainer( e1, e2, msg1, msg2 )
		{;}

		Conn* conn( Eref e, unsigned int funcIndex ) const {
			numIter_++; // For reference counting. Do we need it?
			if ( e.e == e1() )
				return new All2AllConn( funcIndex, this, e.i );
			else
				return new ReverseAll2AllConn( funcIndex, this, e.i );
		}

		bool add( Element* e1, Element* e2 ) {
			All2AllConnEntry ce( e1, e2 );
			c_.push_back( ce );
		}

		unsigned int i1() const {
			return i1_;
		}

		unsigned int i2() const {
			return i2_;
		}

	private:
		unsigned int i1_;
		unsigned int i2_;
};

#endif // _ALL_TO_ALL_CONN_H
