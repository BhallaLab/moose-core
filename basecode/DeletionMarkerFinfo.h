/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DELETION_MARKER_FINFO_H
#define _DELETION_MARKER_FINFO_H

/**
 * Finfo for reporting that host element is about to be deleted
 * Does not participate in anything else.
 */
class DeletionMarkerFinfo: public Finfo
{
		public:
			DeletionMarkerFinfo();

			~DeletionMarkerFinfo()
			{;}

			bool add( 
					Eref e, Eref destElm, const Finfo* destFinfo,
					unsigned int connTainerOption
			) const 
			{
					return 0;
			}
			
			bool respondToAdd(
					Eref e, Eref dest, const Ftype *destType,
					unsigned int& destFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const 
			{
					return 0;
			}

			int msg() const {
				return INT_MAX;
			}

			/**
			 * Call the RecvFunc with the arguments in the string.
			 */
			bool strSet( Eref e, const std::string &s )
					const
					{
							return 0;
					}
			
			/// strGet doesn't work for DeletionMarkerFinfo
			bool strGet( Eref e, std::string &s ) const {
				return 0;
			}

			RecvFunc recvFunc() const {
					return 0;
			}

			// DeletionMarkerFinfo does not allocate any MsgSrc or MsgDest
			// so it does not use this function.
			void countMessages( unsigned int& num )
			{ ; }

			unsigned int syncFuncId() const {
				return 0;
			}

			unsigned int asyncFuncId() const {
				return 0;
			}

			unsigned int proxyFuncId() const {
				return 0;
			}

			bool isTransient() const {
					return 0;
			}

			bool inherit( const Finfo* baseFinfo ) {
				return ftype()->isSameType( baseFinfo->ftype() );
			}

			Finfo* copy() const {
				return new DeletionMarkerFinfo( *this );
			}

			void addFuncVec( const string& cname )
			{;}

			///////////////////////////////////////////////////////
			// Class-specific functions below
			///////////////////////////////////////////////////////

			/**
			 * This function allows us to place a single statically
			 * created DeletionMarkerFinfo wherever it is needed.
			 */
			static DeletionMarkerFinfo* global();

		private:
};

#endif // _DELETION_MARKER_FINFO_H
