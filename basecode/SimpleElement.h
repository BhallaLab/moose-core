/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SIMPLE_ELEMENT_H
#define _SIMPLE_ELEMENT_H

/**
 * The SimpleElement class implements Element functionality in the
 * most common vanilla way. It manages a set of vectors and pointers
 * that keep track of messaging and field information.
 */
class SimpleElement: public Element
{
	public:
#ifdef DO_UNIT_TESTS
			friend void cinfoTest(); // Need to look at src_ and dest_
			friend void msgSrcTest(); // Need to look at src_ and dest_
			friend void msgFinfoTest(); // Need to look at src_ 
			friend void finfoLookupTest(); // to do these tests
			static int numInstances;
#endif
		SimpleElement( const string& name );

		SimpleElement(
				const std::string& name, 
				unsigned int srcSize,
				unsigned int destSize,
				void* data = 0
		);

		/// This cleans up the data_ and finfo_ if needed.
		~SimpleElement();

		const std::string& name( ) const {
				return name_;
		}

		void setName( const std::string& name ) {
				name_ = name;
		}

		const std::string& className( ) const;

		/**
		 * Looks up a Conn entry.
		 */
		vector< Conn >::const_iterator
				lookupConn( unsigned int i ) const;

		/**
		 * Looks up a Conn entry, specialized variant for cases where
		 * we want to modify it.
		 */
		vector< Conn >::iterator lookupVariableConn( unsigned int i );

		/**
		 * This function takes an absolute Conn index 'conn' and
		 * a specified MsgSrc 'src' and returns the appropriate
		 * RecvFunc. It may have to search through the link list of
		 * MsgSrcs to find the correct src and its RecvFunc.
		 * Returns dummyFunc on failure.
		 *
		 * This function is likely to be called by SendTo functions.
		 * There are three likely use cases:
		 * An incoming message is being returned to sender.
		 * A subset of messages are being activated following a search
		 * A local array is maintained and one of them is triggered
		 * Only the last of these seems to need relative indexing,
		 * so we'll keep that in abeyance for now and implement an
		 * absolute send.
		 *
		 */
		RecvFunc lookupRecvFunc( unsigned int src, unsigned int conn )
				const;

		/**
		 * Returns the index of the specified Conn.
		 */
		unsigned int connIndex( const Conn* ) const;

		// Returns size of the conn vector.
		unsigned int connSize( ) const {
				return conn_.size();
		}

		/////////////////////////////////////////////////////////////
		// Src handling functions
		/////////////////////////////////////////////////////////////

		/**
		 * This function returns the iterator to conn_ at the beginning
		 * of the Src range specified by i. Note that we don't need
		 * to know how the Element handles MsgSrcs here.
		 */
		vector< Conn >::const_iterator
				connSrcBegin( unsigned int src ) const;

		/**
		 * This function returns the iterator to conn_ at the end
		 * of the Src range specified by i. End here is in the same
		 * sense as the end() operator on vectors: one past the last
		 * entry. Note that we don't need
		 * to know how the Element handles MsgSrcs here.
		 */
		vector< Conn >::const_iterator
				connSrcEnd( unsigned int src ) const;

		/**
		 * Returns the index of the next src entry on this list of srcs.
		 * Zero if there is none.
		 */
		unsigned int nextSrc( unsigned int src ) const;

		/**
		 * Looks up the RecvFunc placed on the specified src.
		 * This function is a little low-level because it looks
		 * directly into the src_ vector.
		 * It assumes we know exactly which src to use, even if
		 * it is one downstream on the src link list.
		 * The current function is mostly used in the send function.
		 *
		 * Use the lookupRecvFunc function if you only know the 
		 * beginning src index and the Conn index, but do not know
		 * which specific src to use.
		 */
		RecvFunc srcRecvFunc( unsigned int src ) const;

		/**
		 * Returns the number of MsgSrc entries
		 */
		unsigned int srcSize() const {
				return src_.size();
		}

		/////////////////////////////////////////////////////////////
		// Dest handling functions
		/////////////////////////////////////////////////////////////
		/**
		 * This function returns the iterator to conn_ at the beginning
		 * of the Dest range specified by i.
		 */
		vector< Conn >::const_iterator
				connDestBegin( unsigned int dest ) const;

		/**
		 * This function returns the iterator to conn_ at the end
		 * of the Dest range specified by i. End here is in the same
		 * sense as the end() operator on vectors: one past the last
		 * entry.
		 */
		vector< Conn >::const_iterator
				connDestEnd( unsigned int dest ) const;
		
		/**
		 * Returns the number of MsgDest entries
		 */
		unsigned int destSize() const {
				return dest_.size();
		}

		unsigned int insertConn(
				unsigned int src, unsigned int nSrc,
				unsigned int dest, unsigned int nDest );

		void connect( unsigned int myConn, 
				Element* targetElement, unsigned int targetConn );

		void disconnect( unsigned int connIndex );

		void deleteHalfConn( unsigned int connIndex );

		/**
		 * Reports if this element is going to be deleted.
		 */
		bool isMarkedForDeletion() const;

		/**
		 * Puts the death mark on this element.
		 */
		void prepareForDeletion( bool stage );

		unsigned int insertConnOnSrc(
				unsigned int src, FuncList& rf,
				unsigned int dest, unsigned int nDest
		);

		unsigned int insertConnOnDest(
						unsigned int dest, unsigned int nDest );

		void* data() const {
			return data_;
		}

		/**
		 * Regular lookup for Finfo from its name.
		 */
		const Finfo* findFinfo( const string& name );

		/**
		 * Returns finfo ptr associated with specified conn index.
		 * For ordinary finfos, this is a messy matter of comparing
		 * the conn index with the ranges of MsgSrc or MsgDest
		 * entries associated with the finfo. However, this search
		 * happens after the dynamic finfos on the local element.
		 * For Dynamic Finfos, this is fast: it just scans through
		 * all finfos on the local finfo_ vector to find the one that 
		 * has the matching connIndex.
		 */
		const Finfo* findFinfo( unsigned int connIndex ) const;

		unsigned int listFinfos( vector< const Finfo* >& flist ) const;

		void addFinfo( Finfo* f );

		bool isConnOnSrc(
			unsigned int src, unsigned int conn ) const;

		bool isConnOnDest(
			unsigned int dest, unsigned int conn ) const;

	private:
		string name_;

		/**
		 * The conn_ vector contains all the connections emanating
		 * from this Element. Conns belonging to different MsgSrcs
		 * or MsgDests are sequentially ordered, without gaps.
		 * Furthermore, if there are subsets of a MsgSrc that are
		 * going to targets with a given RecvFunc, these too are
		 * sequential. The grouping of such sets is done by putting
		 * extra MsgSrcs in a linked list.
		 * The sequential order also makes it possible to extract the
		 * index of any Conn by referring back to the parent pointer.
		 */
		vector< Conn > conn_; /// All the conns 

		/**
		 * The MsgSrc manages groups of ranges. The MsgSrcs are
		 * pre-allocated and at least the initial set are hard-coded
		 * to refer to specific message groups.
		 */
		vector< MsgSrc > src_;

		/**
		 * The dest_ vector is two ints, specifying the beginning and
		 * end of the group of ranges being used. Here we cannot assume
		 * sequential definitions, so we have to specify the
		 * ranges explicitly.
		 */
		vector< MsgDest > dest_;

		vector< Finfo* > finfo_;
		// Cinfo* cinfo_;
		void* data_;

		// bool isMarkedForDeletion_;

		/**
		 * Scans through all MsgSrcs to find the highest Conn index
		 * used for MsgSrcs.
		 */
		unsigned int lastSrcConnIndex() const;

		/**
		 * Scans through all MsgDests to find the highest Conn index
		 * used for MsgDests.
		 */
		unsigned int lastDestConnIndex() const;

};

#endif // _SIMPLE_ELEMENT_H
