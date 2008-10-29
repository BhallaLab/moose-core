/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _ASYNC_DEST_FINFO_H
#define _ASYNC_DEST_FINFO_H

/**
 * Finfo for handling data to be serialized, usually into an MPI
 * message.
 */
class AsyncDestFinfo: public DestFinfo
{
		public:

			/**
			 * Sets up an AsyncDestFinfo in the usual way.
			 * The RecvFunc here is a little special. It plugs data into
			 * the buffer at the location specified by the connection
			 * index. This is pretty much the same as is done by 
			 * the Ftype specialized versions, that are passed back by
			 * respondToAdd.
			 * But it isn't commonly used and is here for testing
			 * and for reference.
			 */
			AsyncDestFinfo( const string& name, 
				const Ftype *f,
				RecvFunc rfunc,
				const string& doc="",
				unsigned int destIndex = 0 );

			~AsyncDestFinfo()
			{;}

			bool respondToAdd(
					Eref e, Eref dest, const Ftype *destType,
					unsigned int& myFuncId, unsigned int& returnFuncId,
					int& destMsgId, unsigned int& destIndex
			) const;
			
			/**
			 * Returns true only if the other finfo is the same type
			 * For the AsyncDestFinfo this means that every function
			 * argument matches. Then on top of this, the function
			 * copies over the slot indices.
			 */
			bool inherit( const Finfo* baseFinfo ) {
					return 0;
			}

			Finfo* copy() const {
				return new AsyncDestFinfo( *this );
			}

		private:
};

#endif // _ASYNC_DEST_FINFO_H
