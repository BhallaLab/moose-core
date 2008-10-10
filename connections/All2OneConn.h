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
 * This class handles connections where many objects project to
 * a single one. An example is when many instances of ion channels
 * all project to the same gate in the library, that defines their
 * shared kinetic properties.
 * e1 refers to the array of many objects.
 * e2 refers to the single object.
 */
class All2OneConnTainer: public ConnTainer
{
	public:
		All2OneConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		Conn* conn( Eref e, unsigned int funcIndex ) const;
		/*
		Conn* conn( Eref e, unsigned int funcIndex, 
			unsigned int connIndex ) const;
			*/

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
			if ( eI2_ == eIndex )
				return e1numEntries_;
			return 0;
		}

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 * There is of course, precisely one target regardless of the
		 * eIndex.
		 */
		unsigned int numDest( unsigned int eIndex ) const {
			return 1;
		}

		/**
		 * return eIndex of source.
		 * This is a bit messy. I return 0 but could just as well
		 * return AnyIndex.
		 * \todo: Set policy for this.
		 */
		unsigned int eI1() const {
			return 0;
		}

		/**
		 * Return eIndex of target
		 */
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

		/**
		 * Return index to identify which variant of connTainer this is
		 */
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
		unsigned int eI2_;	// eI2 has only the one entry, but could be
			// one entry in an array. So we keep track of the index.
		unsigned int e1numEntries_; // eI1 is all of these.
		unsigned int i1_;
		unsigned int i2_;
};

/**
 * This goes from the many to the one
 * Index is index of source, relevant when sending stuff back.
 * Must be initialized to eIi.
 */
class All2OneConn: public Conn
{
	public:
		All2OneConn( unsigned int funcIndex,
			const All2OneConnTainer* s, unsigned int index )
			: 
				Conn( funcIndex ),
				s_( s ), index_( index )
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
		 * targets. There is only one target, so all this does is to
		 * indicate that the iteration is at an end.
		 */
		void increment() {
			s_ = 0;
		}
		/**
		 * There is only one element, so all this does is indicate that
		 * the iteration is ended.
		 */
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
		const All2OneConnTainer* s_;
		unsigned int index_;	 // Keeps track of e1 element index
};

/**
 * This does serious iteration. It needs to scan back from the
 * single target on element2 to all the possible sources on element1.
 * Needs therefore to know # of sources.
 */
class ReverseAll2OneConn: public Conn
{
	public:
		ReverseAll2OneConn( unsigned int funcIndex,
			const All2OneConnTainer* s, 
			unsigned int index, unsigned int size )
			: Conn( funcIndex ), 
			s_( s ), 
			index_( index ), // Index of e1, ie, originating source?
			size_( size ) // Number of sources
		{;}

		~ReverseAll2OneConn()
		{;}

		Eref target() const {
			return Eref( s_->All2OneConnTainer::e1(), index_ );
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
		const Conn* flip( unsigned int funcIndex ) const {
			return new All2OneConn( funcIndex, s_, index_ );
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

#endif // _ALL2ONE_CONN_H
