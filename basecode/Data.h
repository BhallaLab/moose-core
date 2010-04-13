/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


class Data
{
	public:
		virtual ~Data()
			{;}
		virtual void process( const ProcInfo* p, const Eref& e ) = 0;

		/**
		 *
		 * Lookup function for nested arrays:
		 * 	What if we want to look up two things?
		 * Data* lookup( unsigned int index );
		 * unsigned int numEntries() const;
		 *	Good thing is that this is recursive.
		 *
		 */

		/**
		 * Every Data class must provide a function to initialize its
		 * ClassInfo.
		virtual const Cinfo* initCinfo() = 0;
		 */

#if 0
		/**
		 * Converts object into a binary stream. Returns size.
		 */
		virtual unsigned int serialize( vector< char >& buf ) const;

		/**
		 * Creates object from binary stream.
		 */
		virtual Data* unserialize( vector< char >& buf );
#endif
};
