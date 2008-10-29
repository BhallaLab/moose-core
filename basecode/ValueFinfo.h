/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _VALUE_FINFO_H
#define _VALUE_FINFO_H

/**
 * Finfo for handling data fields that are accessed through get/set
 * functions. Such fields are atomic, that is, they cannot be the
 * base for indirection. This is because the set operation works on
 * the object as a whole. The special use of such fields is if we 
 * want to present an evaluated field: a field whose value is not
 * stored explicitly, but computed (or assigned) on the fly. An example
 * would be the length of a dendritic compartment (a function of its
 * coordinates and those of its parent) or the concentration of a
 * molecule (from the # of molecules and its volume).
 */
class ValueFinfo: public Finfo
{
		public:
			ValueFinfo(
				const string& name,
				const Ftype* f,
				GetFunc get, RecvFunc set,
				const string& doc = ""
			);

			~ValueFinfo()
			{;}

			/**
			 * This operation requires the formation of a dynamic
			 * Finfo to handle the messaging, as Value fields are
			 * not assigned a message src or dest.
			 * \todo: Still to implement.
			 */
			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const ;
			
			bool respondToAdd(
					Eref e, Eref src, const Ftype *srcType,
					unsigned int& srcFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;

			/**
			 * Returns msg number. This is initialized to INT_MAX,
			 * and is used to separate out ValueFinfos for the last
			 * batch of msg numbers. 
			 * Finally it gets assigned by Cinfo::shuffleFinfos which
			 * calls this->countMessages.
			 */
			int msg() const {
				return msg_;
			}

			/**
			 * The Ftype knows how to do this conversion.
			 */
			bool strSet( Eref e, const std::string &s ) const
			{ 
					return ftype()->strSet( e, this, s );
			}
			
			// The Ftype handles this conversion.
			bool strGet( Eref e, std::string &s ) const
			{
					return ftype()->strGet( e, this, s );
			}

			bool isTransient() const {
					return 0;
			}

			/**
			 * Permit override for a field with the same name and type
			 */
			bool inherit( const Finfo* baseFinfo );

			RecvFunc recvFunc() const {
					return set_;
			}

			GetFunc innerGetFunc() const {
					return get_;
			}

			Finfo* copy() const {
				return new ValueFinfo( *this );
			}

			void addFuncVec( const string& cname );

			/**
			 * Returns the identifier for its FuncVec, which handles
			 * its RecvFunc.
			 */
			unsigned int funcId() const {
				return fv_->id();
			}

			/**
			 * The ValueFinfo does not handle any messages itself, so
			 * does not need to allocate any on the parent object.
			 */
			void countMessages( unsigned int& num ) {
				msg_ = -num;
				num++;
			}

			/// Looks at the ftype.
			unsigned int syncFuncId() const {
				return ftype()->syncFuncId();
			}

			/// Looks at the ftype.
			unsigned int asyncFuncId() const {
				return ftype()->asyncFuncId();
			}

			/// Looks at the ftype.
			unsigned int proxyFuncId() const {
				return ftype()->proxyFuncId();
			}

		private:
			GetFunc get_;
			RecvFunc set_;
			FuncVec* fv_;
			int msg_;
};

#endif // _VALUE_FINFO_H
