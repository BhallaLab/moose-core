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
		ArrayElement(
				Id id,
				const std::string& name, 
				void* data,
				unsigned int numSrc, 
				unsigned int numEntries = 0,
				size_t objectSize = 0
		);

		/**
		 * Copies over the name of the ArrayElement, and assigns a
		 * scratch id.
		 * Does not copy the data or the Finfos.
		 * Those are done in the 'innerCopy'
		 * The messages are still more complicated and are done
		 * in other stages of the Copy command.
		 */
		 
		ArrayElement(
			Id id,
			const std::string& name, 
			const unsigned int numSrc,
// 			const vector< Msg >& msg, 
// 			const map< int, vector< ConnTainer* > >& dest,
			const vector< Finfo* >& finfo, 
			void *data, 
			int numEntries, 
			size_t objectSize
		);
		
		ArrayElement( const ArrayElement* orig, Id id = Id() );

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

		/////////////////////////////////////////////////////////////
		// Msg traversal functions, part of API
		/////////////////////////////////////////////////////////////

		/**
		 * msgNum specifies the message.
		 * Returns a Conn* iterator for going through all the targets,
		 * as well as providing lots of collateral information.
		 * Targets can be advanced by increment(), which goes one Element
		 * and its index at a time, or by nextElement(), which goes
		 * one Element at a time even if the Element is an array.
		 * The Conn* must be deleted after use.
		 */
		Conn* targets( int msgNum, unsigned int eIndex ) const;

		/**
		 * finfoName specifies the finfo connecting to these targets.
		 * Returns a Conn* iterator for going through all the targets.
		 * Must be deleted after use.
		 */
		Conn* targets( const string& finfoName, unsigned int eIndex ) const;

		/**
		 * Finds the number of targets to this Msg, either src or dest.
		 * Faster than iterating through the whole lot.
		 */
		unsigned int numTargets( int msgNum ) const;

		/**
		 * Finds the number of targets to this Msg, either src or dest,
		 * on the specified eIndex.
		 * Slow, but faster than iterating through the whole lot if there
		 * are composite messages.
		 */
		unsigned int numTargets( int msgNum, unsigned int eIndex ) const;
		
		/**
		  * Returns the element is of type Array
		  */
		string elementType() const
		{
			return "Array";
		}

		/**
		 * Finds the number of targets to this Finfo.
		 * Faster than iterating through the whole lot.
		 */
		unsigned int numTargets( const string& finfoName ) const;

		/////////////////////////////////////////////////////////////
		// Information functions
		/////////////////////////////////////////////////////////////
		
		/// Computes the memory use by the Element, data and its messages.
		unsigned int getTotalMem() const;

		/// Computes the memory use by the messages.
		unsigned int getMsgMem() const;

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


		/**
		 * Returns data contents of ArrayElement
		 */
		void* data( unsigned int eIndex ) const;

		/**
		 * Returns size of data array. For ArrayElement it can be any
		 * positive integer.
		 */
		unsigned int numEntries() const {
			return numEntries_;
		}

		/////////////////////////////////////////////////////////////
		// Finfo functions
		/////////////////////////////////////////////////////////////


		/**
		 * Regular lookup for Finfo from its name.
		 */
		const Finfo* findFinfo( const string& name );

		/**
		 * Lookup Finfo from its msgNum. Not all Finfos will have a 
		 * msgNum, but any valid msgNum should have a Finfo.
		 */
		const Finfo* findFinfo( int msgNum ) const;

		/**
		 * Special const lookup for Finfo from its name, where the returned
		 * Finfo is limited to the ones already defined in the class
		 * and cannot be an array or other dynamic finfo
		 */
		const Finfo* constFindFinfo( const string& name ) const;

		/**
		 * Returns finfo ptr associated with specified ConnTainer.
		 */
		const Finfo* findFinfo( const ConnTainer* c ) const;

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

		void addExtFinfo( Finfo* f );
		void addFinfo( Finfo* f );
		bool dropFinfo( const Finfo* f );
		void setThisFinfo( Finfo* f );
		const Finfo* getThisFinfo( ) const;

		/////////////////////////////////////////////////////////////
		// Msg handling functions
		/////////////////////////////////////////////////////////////

		/**
		 * Returns a pointer to the specified msg.
		 */
		const Msg* msg( unsigned int msgNum ) const;
		
		Msg* varMsg( unsigned int msgNum );

		/**
		 * Returns a variable pointer to the base msg, in case the 
		 * msgNum indicates one of the later Msgs on the linked list
		 */
		Msg* baseMsg( unsigned int msgNum );

		const vector< ConnTainer* >* dest( int msgNum ) const;

		/**
		 * Scan through dest entries looking for dest msg. Return it if
		 * found. If not found, create a new entry for it and return it. 
		 * This is currently managed by a map indexed by the msgNum.
		 */
		vector< ConnTainer* >* getDest( int msgNum );

		/**
		 * Returns a pointer to the specified msg by looking up the named
		 * Finfo. This may entail construction of a DynamicFinfo, so the
		 * function is not const.
		 * deprecated.
		 */
		// const Msg* msg( const string& fName );

		unsigned int addNextMsg();

		/**
		 * Returns the # of msgs
		 */
		unsigned int numMsg() const;
		
		///////////////////////////////////////////////////////////////
		// Functions for the copy operation. All 5 are virtual
		///////////////////////////////////////////////////////////////
		Element* copy(
				Element* parent,
				const string& newName,
				IdGenerator& idGen ) const
		{ return 0; }
		
		Element* copyIntoArray(
				Id parent,
				const string& newName,
				int n,
				IdGenerator& idGen ) const
		{ return 0; }
		
		bool isDescendant( const Element* ancestor ) const
		{ return 0; }

		Element* innerDeepCopy(
				map< const Element*, Element* >& tree,
				IdGenerator& idGen ) const
		{ return 0; }
		
		Element* innerDeepCopy(
				map< const Element*, Element* >& tree,
				int n,
				IdGenerator& idGen ) const
		{ return 0; }
		
		/*
		void replaceCopyPointers(
					map< const Element*, Element* >& tree,
					vector< pair< Element*, unsigned int > >& delConns );
		void copyMsg( map< const Element*, Element* >& tree );
		*/

		/**
 		* Copies messages from current element to duplicate provided dest is
 		* also on tree.
 		*/
		void copyMessages( Element* dup, 
				map< const Element*, Element* >& origDup,
				bool isArray ) const;

		/**
		 * Copies messages present between current element and globals,
		 * to go between the duplicate and the same globals. Handles
		 * src as well as dest messages.
		 */
		void copyGlobalMessages( Element* dup, bool isArray ) const;

		///////////////////////////////////////////////////////////////
		// Debugging function
		///////////////////////////////////////////////////////////////
		void dumpMsgInfo() const;

		// bool innerCopyMsg( const Conn* c, const Element* orig, Element* dup );
		/**
		 * Overrides the default function provided by the Element. Here we
		 * assign the index to AnyIndex
		 */
		Id id() const;
		
		/**
		 * set values to inter element distance variables
		 */
		void setDistances(double dx, double dy){
			dx_ = dx;
			dy_ = dy;
		}
		
		void setNoOfElements(unsigned int Nx, unsigned int Ny){
			Nx_ = Nx;
			Ny_ = Ny;
		}
		
		void setOrigin(double x, double y){
			xorigin_ = x;
			yorigin_ = y;
		}
		
		double getX(unsigned int index){
			assert(index < numEntries_);
			unsigned int colno = index % Nx_;
			return xorigin_ + dx_*colno;
		}
		
		double getY(unsigned int index){
			assert(index < numEntries_);
			unsigned int colno = index / Nx_;
			return yorigin_ + dy_*colno;
		}
		
	protected:
		Element* innerCopy( IdGenerator& idGen ) const
		{ return 0; }
		
		Element* innerCopy( int n, IdGenerator& idGen ) const
		{ return 0; }

	private:
		/**
		 * Name of element.
		 */
		string name_;

		/**
		 * This stores the field info (Finfo) entries that describe
		 * everything about what the Element does. The Finfo[0] is
		 * special as it also points to the class info, which in turn 
		 * points to the static finfos that define the built-in fields.
		 * The local finfo_ fields are dynamic and are used to extend
		 * the class in various ways.
		 */
		vector< Finfo* > finfo_;

		/**
		 * This stores the actual data contents of the element. Can be
		 * any object.
		 */
		void* data_;

		/**
		 * The Msg manages messages. The Msg vector contains three
		 * sections: the first is for the src, the second for 'next',
		 * and the third for 'dest' entries.
		 * The vector is allocated to the 'src' set when the Element is
		 * initialized. The entries in the src set are hard-coded
		 * by index to refer to specific message groups. 
		 */
		vector< Msg > msg_;

		/**
		 * The destMsg manages pure destination messages. It puts them
		 * into a map to avoid having to store all the possible
		 * locations explicitly. It is not accessed as often, so it
		 * does not need to be a vector.
		 */
		map< int, vector< ConnTainer* > > dest_;

		/**
		 * Index of last entry in 'next_' set of msgs. At initialization,
		 * it indexes the end of the entire msg_ vector, and expands out
		 * as next entries are added, always at the end. After this point
		 * the dests start.
		 */
		 
		 unsigned int numEntries_;

		/**
		 * This is the size of each object in the array.
		 */
		size_t objectSize_;
		
		//createmap specific variables
		unsigned int Nx_, Ny_;
		double dx_, dy_;
		double xorigin_, yorigin_;
		// Deprecated
		// unsigned int destMsgBegin_;
};

#endif // _ARRAY_ELEMENT_H
