/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ARRAY_ELEMENT_H
#define _ARRAY_ELEMENT_H
#include <limits.h>

/**
 * The ArrayElement class implements Element functionality in the
 * most common vanilla way. It manages a set of vectors and pointers
 * that keep track of messaging and field information.
 */
class ArrayElement: public Element
{
	public:
#ifdef DO_UNIT_TESTS
			friend void cinfoTest(); // Need to look at src_ and dest_
			friend void msgSrcTest(); // Need to look at src_ and dest_
			friend void msgFinfoTest(); // Need to look at src_ 
			friend void finfoLookupTest(); // to do these tests
			static int numInstances;
#endif
		ArrayElement( Id id, const string& name );

		ArrayElement(
				Id id,
				const std::string& name, 
				unsigned int srcSize,
				unsigned int destSize,
				void* data = 0,
				unsigned int numEntries = 0,
				size_t objectSize = 0
		);

		// Used in copies.
		ArrayElement( const ArrayElement* orig );
		
		//used in copies
		ArrayElement(
				const std::string& name, 
				const vector< MsgSrc >& src,
				const vector< MsgDest >& dest,
				const vector< Conn >& conn,
				const vector< Finfo* >& finfo,
				void* data,
				unsigned int numEntries,
				size_t objectSize
		);

		/// This cleans up the data_ and finfo_ if needed.
		~ArrayElement();

		const std::string& name( ) const {
				return name_;
		}

		void setName( const std::string& name ) {
				name_ = name;
		}

		const std::string& className( ) const;

		const Cinfo* cinfo() const;

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

		/// Finds the relative index of a conn arriving at this element.
		unsigned int connDestRelativeIndex(
				const Conn& c, unsigned int slot ) const;

		// Finds relative index of conn arriving at this element on
		// the MsgSrc vector identified by slot.
		unsigned int connSrcRelativeIndex(
				const Conn& c, unsigned int slot ) const;

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
		 * Note: this call does NOT follow the linked list of src_ to
		 * the very end. It applies only to the set of Conns that are
		 * attached to a given recvFunc and are therefore on a single
		 * src_ entry.
		 * If you want to follow the linked list, use the nextSrc
		 * function.
		 */
		vector< Conn >::const_iterator
				connSrcEnd( unsigned int src ) const;

		/**
		 * This function does follow the linked list of src_ to the
		 * very end.
		 */
		vector< Conn >::const_iterator
				connSrcVeryEnd( unsigned int src ) const;

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

		/// Computes the memory use by the Element and its messages.
		unsigned int getMsgMem() const;
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
		 * Reports if this element is Global, i.e., should not be copied
		 */
		bool isGlobal() const;

		/**
		 * Puts the death mark on this element.
		 */
		void prepareForDeletion( bool stage );

		unsigned int insertConnOnSrc(
				unsigned int src, FuncList& rf,
				unsigned int dest, unsigned int nDest
		);

		unsigned int insertSeparateConnOnSrc(
			unsigned int src, FuncList& rf,
			unsigned int dest, unsigned int nDest );

		unsigned int insertConnOnDest(
						unsigned int dest, unsigned int nDest );

		void* data() const {
			return data_;
		}

		unsigned int numEntries() const {
			return numEntries_;
		}

		unsigned int index() const {
			return UINT_MAX;
		}
		
		/**
		 * Regular lookup for Finfo from its name.
		 */
		const Finfo* findFinfo( const string& name );

		const Finfo* constFindFinfo( const string& name ) const;

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

		/**
		 * Local finfo_ lookup.
		 */
		const Finfo* localFinfo( unsigned int index ) const;

		/**
		 * Finds all the Finfos associated with this Element,
		 * starting from the local ones and then going to the 
		 * core class ones.
		 * Returns number of Finfos found.
		 */
		unsigned int listFinfos( vector< const Finfo* >& flist ) const;

		/**
		 * Finds the local Finfos associated with this Element.
		 * Note that these are variable. Typically they are Dynamic
		 * Finfos.
		 * Returns number of Finfos found.
		 */
		unsigned int listLocalFinfos( vector< Finfo* >& flist );
		
		/**
		 * Copies Finfos from the SimpleElement to the current Array
		 * Element
		*/
		void CopyFinfosSimpleToArray(const SimpleElement *se);

		void addExtFinfo( Finfo * );
		void addFinfo( Finfo* f );
		bool dropFinfo( const Finfo* f );
		void setThisFinfo( Finfo* f );
		const Finfo* getThisFinfo( ) const;

		bool isConnOnSrc(
			unsigned int src, unsigned int conn ) const;

		bool isConnOnDest(
			unsigned int dest, unsigned int conn ) const;

		///////////////////////////////////////////////////////////////
		// Functions for the copy operation. All 5 are virtual
		///////////////////////////////////////////////////////////////
		Element* copy( Element* parent, const string& newName ) const;
		Element* copyIntoArray( Element* parent, const string& newName, int n ) const;
		bool isDescendant( const Element* ancestor ) const;

		Element* innerDeepCopy( 
						map< const Element*, Element* >& tree ) const;
		Element* innerDeepCopy(
						map< const Element*, Element* >& tree, int n ) const;

		void replaceCopyPointers(
					map< const Element*, Element* >& tree,
					vector< pair< Element*, unsigned int > >& delConns );
		void copyMsg( map< const Element*, Element* >& tree );

		///////////////////////////////////////////////////////////////
		// Debugging function
		///////////////////////////////////////////////////////////////
		void dumpMsgInfo() const;
		
		void setNoOfElements(int Nx, int Ny){
			Nx_ = Nx;
			Ny_ = Ny;
		}
		
		void setDistances(double dx, double dy){
			dx_ = dx;
			dy_ = dy;
		}
		
		void setOrigin(double xorigin, double yorigin){
			xorigin_ = xorigin;
			yorigin_ = yorigin;
		}
		
		void getElementPosition(int& nx, int& ny, const unsigned int index){
			nx = index%Nx_;
			ny = index/Ny_;
		}
		
		void getElementCoordinates(double& xcoord, double& ycoord, const unsigned int index ){
			xcoord = xorigin_ + (index%Nx_)*dx_;
			ycoord = yorigin_ + (index%Ny_)*dy_;
		}

		bool innerCopyMsg( const Conn& c, const Element* orig, Element* dup );
	protected:
		Element* innerCopy() const;
		Element* innerCopy(int n) const;


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

		/**
		 * This void* points to the start of an array of the object
		 * data. Note that it is NOT an array of object data pointers,
		 * but the actual allocated data.
		 */
		void* data_;

		/**
		 * This is the number of allocated entries in the data array.
		 */
		unsigned int numEntries_;

		/**
		 * This is the size of each object in the array.
		 */
		size_t objectSize_;

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
		
		//createmap specific variables
		int Nx_, Ny_;
		double dx_, dy_;
		double xorigin_, yorigin_;

};

#endif // _ARRAY_ELEMENT_H
