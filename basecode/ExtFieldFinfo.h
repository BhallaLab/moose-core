/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _EXT_FIELD_FINFO_H
#define _EXT_FIELD_FINFO_H
/**
 * Finfo used to create and manage arbitrary extended fields.
 * These include data as well as the ability to send and receive
 * messages as if from a regular Value field.
 */
class ExtFieldFinfo: public Finfo
{
		public:
			ExtFieldFinfo( const string& name, const Ftype* type, const string& doc="" )
					: Finfo( name, type, doc )
			{
				val_ = "";
			}
			

			// Assert that the affected conns have been cleaned up
			// before deleteing this.
			~ExtFieldFinfo()
			{;}
			

			/**
			 * This should be almost the regular SrcFinfo::add
			 * operation, since at this point we have acquired
			 * a MsgSrc slot.
			 */
			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const;
			
			/**
			 * Again, this should be similar to the regular
			 * DestFinfo::respondToAdd operation, using the
			 * MsgDest slot.
			 */
			bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;

			/**
			 * Returns a flag for a bad msg.
			 */
			int msg() const {
				return INT_MAX;
			}

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Eref e, const std::string &s ) const {
				return const_cast<ExtFieldFinfo *>(this)->strSet(e, s);
			}
			
			bool strSet( Eref e, const std::string &s ) {
				val_ = s;	
				return true;
			}
			
			// The Ftype handles this conversion.
			bool strGet( Eref e, std::string &s ) const {
				s = val_;
				return true;
			}
			
			/// Public RecvFunc for receiving function args.
			RecvFunc recvFunc() const {
				return recvFunc_;
			}
			
			
			/// Public RecvFunc for receiving requests to send value.
			RecvFunc trigFunc() const {
				return 0;
			}
			
			/**
			 * Internal RecvFunc for passing to Ftype to construct
			 * recvFunc. May be the same as recvFunc for some classes,
			 * such as ValueFinfo or DestFinfo.
			 */
			RecvFunc innerSetFunc() const {
				return setFunc_;
			}
			
			/**
			 * Internal GetFunc for looking up values on object.
			 * Used to get the value used for trigFunc. Passed into
			 * Ftype to build the trigFunc. Often zero
			 * for objects that do not support gets.
			 */
			GetFunc innerGetFunc() const {
				return getFunc_;
			}

			/**
			 * Returns the matching finfo if it is interested.
			 * Reports if there is a match between the field name and
			 * the finfo. This is just like the base Finfo class
			 * operation, but we reimplement the function here
			 * for clarity.
			 * If this is handling a ValueFinfo, it just checks for the
			 * name.
			 * If an ArrayFinfo, it looks up an index as well
			 * If a NestFinfo, it checks for the entire traversal.
			 * All these are done trivially because at creation time
			 * the name_ field is set up so that it has the entire
			 * matching string.
			 * The only issue with this is that it does not permit
			 * variations in the string, such as spaces in index
			 * braces.
			 */
			const Finfo* match( Element* e, const string& n )
			const {
					if ( n == name() )
							return this;
					return 0;
			}

			/**
			 * The ExtFieldFinfo does not handle any messages itself, so
			 * does not need to allocate any on the parent object.
			 */
			void countMessages( unsigned int& num ) {
				;
			}
			
			/// Looks at the original.
			unsigned int syncFuncId() const {
				return origFinfo_->syncFuncId();
			}

			/// Looks at the original.
			unsigned int asyncFuncId() const {
				return origFinfo_->asyncFuncId();
			}

			/// Looks at the original.
			unsigned int proxyFuncId() const {
				return origFinfo_->proxyFuncId();
			}

			/**
			 * The ExtFieldFinfo is one of the few Finfos that has
			 * a true response to this function.
			 */
			bool isTransient() const {
					return 1;
			}

			/////////////////////////////////////////////////////////
			// Here we define the functions that are unique to this
			// class.
			/////////////////////////////////////////////////////////
			
			const Finfo* origFinfo() const {
					return origFinfo_;
			}
			
			unsigned int destIndex() const {
				return destIndex_;
			}
			
			unsigned int srcIndex() const {
				return srcIndex_;
			}

			unsigned int arrayIndex() const {
				return arrayIndex_;
			}

			/**
			 * This returns a pointer to a field specified by the
			 * indirect_ array, as looked up from the data pointer.
			 * Because we deal with void* pointers, this function
			 * must only be called within properly type-protected
			 * functions.
			 */
			void* traverseIndirection( void* data ) const {return 0;}

			/**
			 * This operation makes no sense for the ExtFieldFinfo
			 */
			bool inherit( const Finfo* baseFinfo ) {
				return 0;
			}

			Finfo* copy() const {
				return new ExtFieldFinfo( *this );
			}

			/**
			 * For the ExtFieldFinfo, we should pass in an existing
			 * FuncVec for set, get, and recv. This will need changes
			 * in constructor and associated code. We do not expect to
			 * create a new FuncVec here.
			 */
			void addFuncVec( const string& cname )
			{;}

		private:
			const Finfo* origFinfo_;

			/// Inner RecvFunc from Finfo for handling assignment
			RecvFunc setFunc_;

			/// Inner GetFunc from Finfo, gets value if applicable.
			GetFunc getFunc_;

			/// Public RecvFunc for receiving function args.
			RecvFunc recvFunc_;

			/// Public RecvFunc for receiving requests to send value.
			RecvFunc trigFunc_;

			unsigned int arrayIndex_;
			unsigned int srcIndex_;
			unsigned int destIndex_;
			// vector< IndirectType > indirect_;
			string val_;
};

/**
 * This function looks up the ExtFieldFonfo matching the incoming Conn
 */
//extern const ExtFieldFinfo* getDF( const Conn* );

#endif // _EXT_FIELD_FINFO_H
