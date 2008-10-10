/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MANY2MANY_CONN_H
#define _MANY2MANY_CONN_H


class Many2ManyConnTainer: public ConnTainer
{
	public:
		Many2ManyConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 = 0, unsigned int i2 = 0 );

		Conn* conn( Eref e, unsigned int funcIndex ) const;

		bool add( Element* e1, Element* e2 );

		/**
		 * Returns the number of targets on this ConnTainer
		 */
		unsigned int size() const {
			return entries_.nEntries();
		}

		/**
		 * Returns the number of sources coming to the specified
		 * eIndex,
		 */
		unsigned int numSrc( unsigned int eIndex ) const;

		/**
		 * Returns the number of targets originating from the specified
		 * eIndex, on this ConnTainer.
		 */
		unsigned int numDest( unsigned int eIndex ) const;

		unsigned int eI1() const { // Irrelevant
			return 0;
		}

		unsigned int eI2() const { // Irrelevant
			return 0;
		}

		unsigned int i1() const { // Irrelevant
			return i1_;
		}

		unsigned int i2() const { // Irrelevant
			return 0;
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
			return Many2Many;
		}

		bool addToConnTainer( 
			unsigned int srcIndex, 
			unsigned int destIndex, unsigned int index );

		///////////////////////////////////////////////////////////
		// class-specific functions
		///////////////////////////////////////////////////////////
		/**
		 * Passes back row of entries as a pointer. Returns the number of 
		 * entries in the row. Also passes back the
		 * corresponding eIndex of the targets. All are of course on 
		 * the target ArrayElement.
		 * This is a fast operation and just returns a pointer into
		 * existing internal data structures.
		 */
		unsigned int getRow( unsigned int rowNum, 
			const unsigned int** index, const unsigned int** eIndex ) const;

		/**
		 * Passes back a column of entries and their eIndices in the
		 * provided vectors. This is a slow operation and has to munge
		 * through the entire sparse matrix to build up the vectors.
		 * Returns number of entries.
		 */
		unsigned int getColumn( unsigned int i, 
			vector< unsigned int >& index,
			vector< unsigned int >& eIndex ) const;
		
	private:
		/**
		 * Here we store the index of the entry at the target.
		 * The src is given by the row, the dest by the column.
		 */
		SparseMatrix< unsigned int > entries_;
		unsigned int i1_;
};

class Many2ManyConn: public Conn
{
	public:
		Many2ManyConn( unsigned int funcIndex,
			const Many2ManyConnTainer* s, unsigned int eIndex )
			: Conn( funcIndex ), s_( s ), srcEindex_( eIndex ), i_( 0 )
		{ 
			targetElm_ = s_->Many2ManyConnTainer::e2();
			size_ = s->getRow( eIndex, &tgtIndexIter_, &tgtEindexIter_ );
		}

		~Many2ManyConn()
		{;}

		Eref target() const {
			return Eref( targetElm_, *tgtEindexIter_ );
		}

		unsigned int targetIndex() const {
			return *tgtIndexIter_;
		}
		int targetMsg() const {
			return s_->Many2ManyConnTainer::msg2();
		}
		Eref source() const {
			return Eref( s_->Many2ManyConnTainer::e1(), srcEindex_ );
		}
		unsigned int sourceIndex() const {
			return s_->Many2ManyConnTainer::i1();
		}
		int sourceMsg() const {
			return s_->Many2ManyConnTainer::msg1();
		}
		void* data() const {
			return s_->Many2ManyConnTainer::e2()->data( *tgtEindexIter_ );
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			++tgtEindexIter_;
			++tgtIndexIter_;
			++i_;
		}
		void nextElement() {
			i_ = size_;
		}
		bool good() const {
			return ( i_ < size_ );
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
		const Many2ManyConnTainer* s_;
		unsigned int srcEindex_;
		const unsigned int* tgtEindexIter_;
		const unsigned int* tgtIndexIter_;
		unsigned int i_;	 // Keeps track of position in row vector of
							// target eIndex and index.

		unsigned int size_;	 	// Keeps track of e2 element size
		Element* targetElm_;
};


/**
 * This class is used for reverse traversal. This is highly discouraged
 * for Many2Many as it is very slow. We expect synaptic info to
 * go forward, and reverse traversal should be rare. If in some other
 * context we have lots of reverse traversal we should use a different
 * form of Many2Many that has full array or similar lookup.
 */
class ReverseMany2ManyConn: public Conn
{
	public:
		ReverseMany2ManyConn( unsigned int funcIndex,
			const Many2ManyConnTainer* s, unsigned int eIndex )
			: Conn( funcIndex ), s_( s ), srcEindex_( eIndex ), i_( 0 )
		{ 
			targetElm_ = s_->Many2ManyConnTainer::e2();
			size_ = s->getColumn( eIndex, tgtIndex_, tgtEindex_ );
		}

		~ReverseMany2ManyConn()
		{;}

		Eref target() const {
			return Eref( targetElm_, tgtEindex_[i_] );
		}

		unsigned int targetIndex() const {
			return tgtIndex_[i_];
		}
		int targetMsg() const {
			return s_->Many2ManyConnTainer::msg1();
		}
		Eref source() const {
			return Eref( s_->Many2ManyConnTainer::e2(), srcEindex_ );
		}
		unsigned int sourceIndex() const {
			return s_->Many2ManyConnTainer::i2();
		}
		int sourceMsg() const {
			return s_->Many2ManyConnTainer::msg2();
		}
		void* data() const {
			return s_->Many2ManyConnTainer::e1()->data( tgtEindex_[i_]);
		}

		/**
		 * increment() updates internal counter, used in iterating through
		 * targets.
		 */
		void increment() {
			++i_;
		}
		void nextElement() {
			i_ = size_;
		}
		bool good() const {
			return ( i_ < size_ );
		}

		/**
		 * Returns a Conn with e1 and e2 flipped so that return messages
		 * traverse back with the correct args.
		 */
		const Conn* flip( unsigned int funcIndex ) const {
			return new Many2ManyConn( funcIndex, s_, tgtEindex_[ i_ ]  );
		}

		const ConnTainer* connTainer() const {
			return s_;
		}

		bool isDest() const  {
			return 1;
		}

	private:
		const Many2ManyConnTainer* s_;
		unsigned int srcEindex_;
		vector< unsigned int > tgtEindex_;
		vector< unsigned int > tgtIndex_;
		unsigned int i_;	 // Keeps track of position in row vector of
							// target eIndex and index.

		unsigned int size_;	 	// Keeps track of e2 element size
		Element* targetElm_;
};
#endif // _MANY2MANY_CONN_H
