/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SIMPLE_CONN_H
#define _SIMPLE_CONN_H


class SimpleConnTainer: public ConnTainer
{
	public:
		SimpleConnTainer( Element* e1, Element* e2, 
			int msg1, int msg2,
			unsigned int eI1 = 0, unsigned int eI2 = 0,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		SimpleConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		Conn* conn( Eref e, unsigned int funcIndex ) const;

		bool add( Element* e1, Element* e2 );

		/**
		 * Returns the number of targets on this ConnTainer
		 */
		unsigned int size() const {
			return 1;
		}

		/**
		 * Returns the number of sources coming to the specified
		 * eIndex,
		 */
		unsigned int numSrc( unsigned int eIndex ) const {
			return ( eIndex == eI2_ );
		}

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		unsigned int numDest( unsigned int eIndex ) const {
			return ( eIndex == eI1_ );
		}

		unsigned int eI1() const {
			return eI1_;
		}

		unsigned int eI2() const {
			return eI2_;
		}

		unsigned int i1() const {
			return i1_;
		}

		unsigned int i2() const {
			return i2_;
		}

		/**
 		 * Creates a duplicate ConnTainer for message(s) between 
 		 * new elements e1 and e2,
 		 * It checks the original version for which msgs to put the new
 		 * one on. e1 must be the new source element.
 		 * Returns the new ConnTainer on success, otherwise 0.
 		*/
		ConnTainer* copy( Element* e1, Element* e2, bool isArray ) const;

		unsigned int option() const {
			return Simple;
		}

		bool addToConnTainer( 
			unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index )
		{
			return 0;
		}
		
	private:
		unsigned int eI1_;
		unsigned int eI2_;
		unsigned int i1_;
		unsigned int i2_;
};

class SimpleConn: public Conn
{
	public:
		SimpleConn( unsigned int funcIndex, const SimpleConnTainer* s )
			: Conn( funcIndex), s_( s )
		{;}

		~SimpleConn()
		{;}

		Eref target() const {
			return Eref( s_->SimpleConnTainer::e2(), s_->SimpleConnTainer::eI2() );
		}
		unsigned int targetIndex() const {
			return s_->SimpleConnTainer::i2();
		}
		int targetMsg() const {
			return s_->SimpleConnTainer::msg2();
		}
		Eref source() const {
			return Eref( s_->SimpleConnTainer::e1(), s_->SimpleConnTainer::eI1() );
		}
		unsigned int sourceIndex() const {
			return s_->SimpleConnTainer::i1();
		}
		int sourceMsg() const {
			return s_->SimpleConnTainer::msg1();
		}
		void* data() const {
			return s_->SimpleConnTainer::e2()->data( s_->SimpleConnTainer::eI2() );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets. Since we have a single entry in the SimpleConn, all
		 * this has to do is to invalidate further good() calls.
		 */
		void increment() {
			s_ = 0;
		}
		void nextElement() {
			s_ = 0;
		}
		bool good() const {
			return ( s_ != 0 );
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const;

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 0;
		}

	private:
		const SimpleConnTainer* s_;
};

class ReverseSimpleConn: public Conn
{
	public:
		ReverseSimpleConn( 
			unsigned int funcIndex, const SimpleConnTainer* s )
			: Conn( funcIndex ), s_( s ) 
		{;}

		~ReverseSimpleConn()
		{;}

		Eref target() const {
			return Eref( s_->SimpleConnTainer::e1() , s_->SimpleConnTainer::eI1() );
		}
		unsigned int targetEindex() const {
			return s_->SimpleConnTainer::eI1();
		}
		unsigned int targetIndex() const {
			return s_->SimpleConnTainer::i1();
		}
		int targetMsg() const {
			return s_->SimpleConnTainer::msg1();
		}
		Eref source() const {
			return Eref( s_->SimpleConnTainer::e2(), s_->SimpleConnTainer::eI2() );
		}
		unsigned int sourceIndex() const {
			return s_->SimpleConnTainer::i2();
		}
		int sourceMsg() const {
			return s_->SimpleConnTainer::msg2();
		}
		void* data() const {
			return s_->e1()->data( s_->SimpleConnTainer::eI1() );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			s_ = 0;
		}

		void nextElement() {
			s_ = 0;
		}

		bool good() const {
			return ( s_ != 0 );
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const {
			return new SimpleConn( funcIndex, s_ );
		}

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 1;
		}

	private:
		const SimpleConnTainer* s_;
};

#endif // _SIMPLE_CONN_H
