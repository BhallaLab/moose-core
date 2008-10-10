/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE2ONE_MAP_CONN_H
#define _ONE2ONE_MAP_CONN_H

/**
 * This class handles connections where each entry in an array connects
 * to a matching entry in another array. For example, if we have set up
 * a prototype cell model where A->B, then the array version of the cell
 * model will have A[i]->B[i] for all i. This situation is handled by
 * the One2OneMap.
 */

class One2OneMapConnTainer: public ConnTainer
{
	public:
		/**
		 * Constructor for One2OneMapConnTainer.
		 * This is a bit unusual because it does a lot of work.
		 * Scans through all the dests to fill up the i2_ vector that
		 * identifies each message as it arrives at the dest. It queries
		 * each e2 entry for its numTargets.
		 */
		One2OneMapConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0 );

		Conn* conn( Eref e, unsigned int funcIndex ) const;

		bool add( Element* e1, Element* e2 );

		/**
		 * Returns the number of targets on this ConnTainer
		 */
		unsigned int size() const {
			return i2_.size();
		}

		/**
		 * Returns the number of targets on this ConnTainer, starting from
		 * the specified eIndex.
		 */
		unsigned int size( unsigned int eIndex ) const {
			if ( eIndex < i2_.size() );
				return 1;
			return 0;
		}

		/**
		 * Returns the number of sources coming to the specified eIndex,
		 */
		unsigned int numSrc( unsigned int eIndex ) const {
			if ( eIndex < i2_.size() );
				return 1;
			return 0;
		}

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		unsigned int numDest( unsigned int eIndex ) const {
			if ( eIndex < i2_.size() );
				return 1;
			return 0;
		}

		unsigned int eI1() const {
			return 0;
		}

		unsigned int eI2() const {
			return 0;
		}

		unsigned int i1() const {
			return i1_;
		}

		const vector< unsigned int >& i2() const {
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
			return One2OneMap;
		}

		/**
		 * Cannot add more messages to a One2OneMap: already goes to its
		 * exclusive target.
		 */
		bool addToConnTainer( 
			unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index )
		{
			return 0;
		}
		
	private:
		unsigned int i1_; // We don't really worry about this.
		vector< unsigned int > i2_;
};

class One2OneMapConn: public Conn
{
	public:
		One2OneMapConn( unsigned int funcIndex, 
			const One2OneMapConnTainer* s, unsigned int index )
			: Conn( funcIndex ), s_( s ), index_( index )
		{
			assert ( index < s->size() );
		}

		~One2OneMapConn()
		{;}

		Eref target() const {
			return Eref( s_->One2OneMapConnTainer::e2(), index_ );
		}
		unsigned int targetIndex() const {
			return s_->One2OneMapConnTainer::i2()[ index_ ]; 
		}
		int targetMsg() const {
			return s_->One2OneMapConnTainer::msg2();
		}
		Eref source() const {
			return Eref( s_->One2OneMapConnTainer::e1(), index_ );
		}

		// This is a cop-out. But we should not really use i1().
		unsigned int sourceIndex() const {
			return s_->One2OneMapConnTainer::i1();
		}
		int sourceMsg() const {
			return s_->One2OneMapConnTainer::msg1();
		}
		void* data() const {
			return s_->One2OneMapConnTainer::e2()->data( index_ );
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
		const Conn* flip( unsigned int funcIndex ) const;

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 0;
		}

	private:
		const One2OneMapConnTainer* s_;
		unsigned int index_;	 // Keeps track of e2 element index
};

class ReverseOne2OneMapConn: public Conn
{
	public:
		ReverseOne2OneMapConn( unsigned int funcIndex, 
			const One2OneMapConnTainer* s, unsigned int index )
			: Conn( funcIndex ), s_( s ), index_( index ) 
		{;}

		~ReverseOne2OneMapConn()
		{;}

		Eref target() const {
			return Eref( s_->One2OneMapConnTainer::e1() , index_ );
		}
		unsigned int targetEindex() const {
			return s_->One2OneMapConnTainer::eI1();
		}
		unsigned int targetIndex() const {
			return s_->One2OneMapConnTainer::i1();
		}
		int targetMsg() const {
			return s_->One2OneMapConnTainer::msg1();
		}
		Eref source() const {
			return Eref( s_->One2OneMapConnTainer::e2(), index_ );
		}
		unsigned int sourceIndex() const {
			return s_->One2OneMapConnTainer::i2()[ index_ ];
		}
		int sourceMsg() const {
			return s_->One2OneMapConnTainer::msg2();
		}
		void* data() const {
			return s_->e1()->data( index_ );
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
			return new One2OneMapConn( funcIndex, s_, index_ );
		}

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 1;
		}

	private:
		const One2OneMapConnTainer* s_;
		unsigned int index_; // Keeps track of the element index
};

#endif // _ONE2ONE_MAP_CONN_H
