/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE2ALL_CONN_H
#define _ONE2ALL_CONN_H


/**
 * This class handles connections where a single object projects to
 * every entry in an array. An example is a simple object with an
 * array object child, or the scheduler tick going to all compartments
 * on a cell.
 */
class One2AllConnTainer: public ConnTainer
{
	public:
		One2AllConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		Conn* conn( Eref e, unsigned int funcIndex ) const;

		bool add( Element* e1, Element* e2 );

		/**
		 * Returns the number of targets on this ConnTainer
		 */
		unsigned int size() const {
			return e2numEntries_;
		}

		/**
		 * Returns the number of sources coming to the specified eIndex,
		 */
		unsigned int numSrc( unsigned int eIndex ) const {
			return 1;
		}

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		unsigned int numDest( unsigned int eIndex ) const {
			if ( eI1_ == eIndex )
				return e2numEntries_;
			return 0;
		}

		unsigned int eI1() const {
			return eI1_;
		}

		unsigned int eI2() const {
			return 0;
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
			return One2All;
		}

		/**
		 * Cannot add more messages to a One2All: already goes to all
		 * possible targets.
		 */
		bool addToConnTainer( 
			unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index )
		{
			return 0;
		}
		
	private:
		unsigned int eI1_;
		unsigned int e2numEntries_;
		unsigned int i1_;
		unsigned int i2_;
};

class One2AllConn: public Conn
{
	public:
		One2AllConn( unsigned int funcIndex, 
			const One2AllConnTainer* s, unsigned int index )
			: Conn( funcIndex ), s_( s ), index_( index ), size_( s->size() )
		{;}

		~One2AllConn()
		{;}

		Eref target() const {
			return Eref( s_->One2AllConnTainer::e2(), index_ );
		}
		unsigned int targetIndex() const {
			return s_->One2AllConnTainer::i2();
		}
		int targetMsg() const {
			return s_->One2AllConnTainer::msg2();
		}
		Eref source() const {
			return Eref( s_->One2AllConnTainer::e1(), s_->One2AllConnTainer::eI1() );
		}
		unsigned int sourceIndex() const {
			return s_->One2AllConnTainer::i1();
		}
		int sourceMsg() const {
			return s_->One2AllConnTainer::msg1();
		}
		void* data() const {
			return s_->One2AllConnTainer::e2()->data( index_ );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			++index_;
		}
		void nextElement() {
			index_ = size_;
		}
		bool good() const {
			return ( index_ < size_ );
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
		const One2AllConnTainer* s_;
		unsigned int index_;	 // Keeps track of e2 element index
		unsigned int size_;	 	// Keeps track of e2 element size
};

class ReverseOne2AllConn: public Conn
{
	public:
		ReverseOne2AllConn( unsigned int funcIndex, 
			const One2AllConnTainer* s, unsigned int index )
			: Conn( funcIndex ), s_( s ), index_( index ) 
		{;}

		~ReverseOne2AllConn()
		{;}

		Eref target() const {
			return Eref( s_->One2AllConnTainer::e1() , s_->One2AllConnTainer::eI1() );
		}
		unsigned int targetEindex() const {
			return s_->One2AllConnTainer::eI1();
		}
		unsigned int targetIndex() const {
			return s_->One2AllConnTainer::i1() + index_;
		}
		int targetMsg() const {
			return s_->One2AllConnTainer::msg1();
		}
		Eref source() const {
			return Eref( s_->One2AllConnTainer::e2(), index_ );
		}
		unsigned int sourceIndex() const {
			return s_->One2AllConnTainer::i2();
		}
		int sourceMsg() const {
			return s_->One2AllConnTainer::msg2();
		}
		void* data() const {
			return s_->e1()->data( s_->One2AllConnTainer::eI1() );
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
			return new One2AllConn( funcIndex, s_, index_ );
		}

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 1;
		}

	private:
		const One2AllConnTainer* s_;
		unsigned int index_; // Keeps track of the e2 element index
};

// Some temporary typedefs while I think about implementations
typedef SimpleConnTainer One2ManyConnTainer;
typedef SimpleConnTainer Many2OneConnTainer;
typedef SimpleConnTainer Many2AllConnTainer;
typedef SimpleConnTainer All2ManyConnTainer;
typedef SimpleConnTainer All2AllMapConnTainer;

#endif // _ONE2ALL_CONN_H
