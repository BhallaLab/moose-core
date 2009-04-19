/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _PROXY_ELEMENT_H
#define _PROXY_ELEMENT_H

typedef void( *ProxyFunc )( const Conn*, const void*, Slot );

/**
 * The ProxyElement class is a minimalist stand-in for an off-node
 * Element. It manages only one Msg and knows its PostMaster, 
 * is all. The rest of the stuff it refers ruthlessly to the original
 * Element on the original node.
 */
class ProxyElement: public Element
{
	public:
		ProxyElement( Id id, unsigned int node, 
			unsigned int proxyFuncId, unsigned int numEntries );

		/// Nothing much to do here.
		~ProxyElement() {
			;
		}

		const std::string& name( ) const {
			static string ret = "proxy";
			// Should look up original on postmaster, or
			// better, the request should not have come here but
			// gone to the original node.
			return ret;
		}

		void setName( const std::string& name ) {
				;
		}

		const std::string& className( ) const {
			static string ret = "proxy";
			return ret;
		}

		const Cinfo* cinfo() const {
			return 0;
		}

		/////////////////////////////////////////////////////////////
		// Hack here to deal with proxy msgs
		/////////////////////////////////////////////////////////////
		void sendData( unsigned int funcIndex, const char* data,
			unsigned int eIndex );

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
		Conn* targets( int msgNum, unsigned int eIndex ) const 
		{
			return 0;
		}

		/**
		 * finfoName specifies the finfo connecting to these targets.
		 * Returns a Conn* iterator for going through all the targets.
		 * Must be deleted after use.
		 */
		Conn* targets( const string& finfoName, unsigned int eIndex ) const
		{
			return 0;
		}

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
		unsigned int numTargets( int msgNum, unsigned int eIndex ) const {
			return 0;
		}

		/**
		 * Finds the number of targets to this Finfo.
		 * Faster than iterating through the whole lot.
		 */
		unsigned int numTargets( const string& finfoName ) const {
			return 0;
		}
		
		/**
		  * Returns the type of the element. Should ideally be an enum,
		  * or just do a dynamic_cast.
		  */
		virtual string elementType() const
		{
			return "Proxy";
		}

		/////////////////////////////////////////////////////////////
		// Information functions
		/////////////////////////////////////////////////////////////
		
		/// Computes the memory use by the Element, data and its messages.
		unsigned int getTotalMem() const {
			return 0;
		}

		/// Computes the memory use by the messages.
		unsigned int getMsgMem() const {
			return 0;
		}

		/**
		 * Reports if this element is going to be deleted.
		 */
		bool isMarkedForDeletion() const {
			return 0;
		}

		/**
		 * Reports if this element is Global, i.e., should not be copied
		 */
		bool isGlobal() const {
			return 0;
		}

		/**
		 * Puts the death mark on this element.
		 */
		void prepareForDeletion( bool stage ) {
			;
		}


		/**
		 * Returns data contents of ProxyElement. 
		 * This simply refers to the postmaster data for the original
		 * node for the proxy.
		 */
		void* data( unsigned int eIndex ) const;

		/**
		 * Returns size of data array. For ProxyElement it is always 1.
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
		const Finfo* findFinfo( const string& name ) {
			return 0;
		}

		/**
		 * Lookup Finfo from its msgNum. Not all Finfos will have a 
		 * msgNum, but any valid msgNum should have a Finfo.
		 */
		const Finfo* findFinfo( int msgNum ) const {
			return 0;
		}

		/**
		 * Special const lookup for Finfo from its name, where the returned
		 * Finfo is limited to the ones already defined in the class
		 * and cannot be an array or other dynamic finfo
		 */
		const Finfo* constFindFinfo( const string& name ) const {
			return 0;
		}

		/**
		 * Returns finfo ptr associated with specified ConnTainer.
		 */
		const Finfo* findFinfo( const ConnTainer* c ) const {
			return 0;
		}

		/**
		 * Local finfo_ lookup.
		 */
		const Finfo* localFinfo( unsigned int index ) const {
			return 0;
		}

		/**
		 * Finds all the Finfos associated with this Element,
		 * starting from the local ones and then going to the 
		 * core class ones.
		 * Returns number of Finfos found.
		 */
		unsigned int listFinfos( vector< const Finfo* >& flist ) const {
			return 0;
		}

		/**
		 * Finds the local Finfos associated with this Element.
		 * Note that these are variable. Typically they are Dynamic
		 * Finfos.
		 * Returns number of Finfos found.
		 */
		unsigned int listLocalFinfos( vector< Finfo* >& flist ) {
			return 0;
		}

		void addExtFinfo( Finfo* f ) {
			;
		}
		void addFinfo( Finfo* f ) {
			;
		}
		bool dropFinfo( const Finfo* f ) {
			return 0;
		}
		void setThisFinfo( Finfo* f ) {
			;
		}
		const Finfo* getThisFinfo( ) const {
			return 0;
		}

		/////////////////////////////////////////////////////////////
		// Msg handling functions
		/////////////////////////////////////////////////////////////

		/**
		 * Returns a pointer to the specified msg.
		 */
		const Msg* msg( unsigned int msgNum ) const {
			return &msg_;
		}
		
		Msg* varMsg( unsigned int msgNum ) {
			return &msg_;
		}

		Msg* baseMsg( unsigned int msgNum ) {
			return &msg_;
		}

		const vector< ConnTainer* >* dest( int msgNum ) const {
			return 0;
		}

		/**
		 * Scan through dest entries looking for dest msg. Return it if
		 * found. If not found, create a new entry for it and return it. 
		 * This is currently managed by a map indexed by the msgNum.
		 */
		vector< ConnTainer* >* getDest( int msgNum ) {
			return 0;
		}

		/**
		 * Create a new 'next' msg entry and return its index
		 * Cannot handle this yet.
		 */
		unsigned int addNextMsg() {
			return 1;
		}

		/**
		 * Returns the # of msgs
		 */
		unsigned int numMsg() const {
			return 1;
		}
		
		///////////////////////////////////////////////////////////////
		// Functions for the copy operation. All 5 are virtual, and 
		// none should be coming to the Proxy.
		///////////////////////////////////////////////////////////////
		Element* copy(
				Element* parent,
				const string& newName,
				IdGenerator& idGen ) const
		{
			return 0;
		}
		
		Element* copyIntoArray(
				Id parent,
				const string& newName,
				int n,
				IdGenerator& idGen ) const
		{
			return 0;
		}

		bool isDescendant( const Element* ancestor ) const {
			return 0;
		}

		Element* innerDeepCopy(
				map< const Element*, Element* >& tree,
				IdGenerator& idGen ) const 
		{
			return 0;
		}
		
		Element* innerDeepCopy(
				map< const Element*, Element* >& tree,
				int n,
				IdGenerator& idGen ) const
		{
			return 0;
		}
		
		/**
 		* Copies messages from current element to duplicate provided dest is
 		* also on tree.
		* This is nasty. Yet to figure out how to handle this sort of thing
		* for proxies and for messages going off node.
 		*/
		void copyMessages( Element* dup, 
			map< const Element*, Element* >& origDup, bool isArray ) const {
			;
		}

		void copyGlobalMessages( Element* dup, bool isArray ) const {
			;
		}
		
		///////////////////////////////////////////////////////////////
		// Debugging function
		///////////////////////////////////////////////////////////////
		void dumpMsgInfo() const {
			;
		}

	protected:
		Element* innerCopy( IdGenerator& idGen ) const {
			return 0;
		}
		Element* innerCopy( int n, IdGenerator& idGen ) const {
			return 0;
		}

	private:
		Msg msg_;			/// Handles messages to actual targets.
		unsigned int node_;	/// Node of original Element.

		/**
		 * Vector of ProxyFuncs, aligned with the recvFuncs of the
		 * target object. Each ProxyFunc converts the serialized
		 * stream from the postMaster to arguments and issues it to the
		 * appropriate Send command for the message. Needs the funcIndex
		 * to identify the correct funcs.
		 */
		const FuncVec* proxyVec_; /// Vector of proxy Funcs

		/**
		 * Size of element array handled by original.
		 */
		unsigned int numEntries_;
};

#endif // _PROXY_ELEMENT_H
