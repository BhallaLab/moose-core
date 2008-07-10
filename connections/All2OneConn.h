/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ALL2ONE_CONN_H
#define _ALL2ONE_CONN_H


/**
 * This class handles connections where a single object projects to
 * every entry in an array. An example is a simple object with an
 * array object child, or the scheduler tick going to all compartments
 * on a cell.
 */
class All2OneConnTainer: public ConnTainer
{
	public:
		All2OneConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		Conn* conn( unsigned int eIndex, bool isReverse ) const;
		Conn* conn( unsigned int eIndex, bool isReverse,
			unsigned int connIndex ) const;

		bool add( Element* e1, Element* e2 );

		/**
		 * Returns the number of targets on this ConnTainer
		 */
		unsigned int size() const {
			return 1;
		}

		/**
		 * Returns the number of sources coming to the specified eIndex,
		 */
		unsigned int numSrc( unsigned int eIndex ) const {
			if ( eI2_ == eIndex );
				return e1numEntries_;
			return 0;
		}

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		unsigned int numDest( unsigned int eIndex ) const {
			return 1;
		}

		unsigned int eI1() const {
			return 0;
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
			return All2One;
		}

		/**
		 * Cannot add more messages to a All2One: already goes to all
		 * possible targets.
		 */
		bool addToConnTainer( 
			unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index )
		{
			return 0;
		}
		
	private:
		unsigned int eI2_;
		unsigned int e1numEntries_;
		unsigned int i1_;
		unsigned int i2_;
};

class All2OneConn: public Conn
{
	public:
		All2OneConn( const All2OneConnTainer* s, unsigned int index )
			: s_( s ), index_( index )
		{;}

		~All2OneConn()
		{;}

		Eref target() const {
			return Eref( s_->All2OneConnTainer::e2(), s_->All2OneConnTainer::eI2() );
		}
		unsigned int targetIndex() const {
			return s_->All2OneConnTainer::i2();
		}
		int targetMsg() const {
			return s_->All2OneConnTainer::msg2();
		}
		Eref source() const {
			return Eref( s_->All2OneConnTainer::e1(), index_ );
		}
		unsigned int sourceIndex() const {
			return s_->All2OneConnTainer::i1();
		}
		int sourceMsg() const {
			return s_->All2OneConnTainer::msg1();
		}
		void* data() const {
			return s_->All2OneConnTainer::e2()->data( s_->All2OneConnTainer::eI2() );
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
		const Conn* flip() const;

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 0;
		}

	private:
		const All2OneConnTainer* s_;
		unsigned int index_;	 // Keeps track of e2 element index
		
};

class ReverseAll2OneConn: public Conn
{
	public:
		ReverseAll2OneConn( const All2OneConnTainer* s, unsigned int index )
			: s_( s ), index_( index ) 
		{;}

		~ReverseAll2OneConn()
		{;}

		Eref target() const {
			return Eref( s_->All2OneConnTainer::e1() , index_ );
		}
		unsigned int targetEindex() const {
			return s_->All2OneConnTainer::eI1();
		}
		unsigned int targetIndex() const {
			return s_->All2OneConnTainer::i1();
		}
		int targetMsg() const {
			return s_->All2OneConnTainer::msg1();
		}
		Eref source() const {
			return Eref( s_->All2OneConnTainer::e2(), s_->All2OneConnTainer::eI2() );
		}
		unsigned int sourceIndex() const {
			return s_->All2OneConnTainer::i2();
		}
		int sourceMsg() const {
			return s_->All2OneConnTainer::msg2();
		}
		void* data() const {
			return s_->e1()->data( index_ );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			++index_;
		}

		void nextElement() {
			index_ = size_ ;
		}

		bool good() const {
			return ( index_ < size_ );
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip() const {
			return new All2OneConn( s_, index_ );
		}

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 1;
		}

	private:
		const All2OneConnTainer* s_;
		unsigned int index_; // Keeps track of the e2 element index
		unsigned int size_;
};

// Some temporary typedefs while I think about implementations
typedef SimpleConnTainer One2ManyConnTainer;
typedef SimpleConnTainer Many2OneConnTainer;
typedef SimpleConnTainer Many2AllConnTainer;
typedef SimpleConnTainer All2ManyConnTainer;
typedef SimpleConnTainer All2AllMapConnTainer;

#endif // _ALL2ONE_CONN_H
