/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This class is a parallel version of SynChan class.
 * This file is compiled only when the parallelizatin flag is enabled.
 * This class derives from SynChan. 
 *
 * 
 * This class refers to the base class, SynChan, for all functionality. 
 * Parallel moose parser would require overriding of some of the base class functionality. 
 * Such functions would be overridden in this class. 
 *
 * 
 */


#ifndef _ParSynChan_h
#define _ParSynChan_h

class ParSynChan : protected SynChan
{

	public:
		ParSynChan();
		static void reinitFunc( const Conn& c, ProcInfo p );
		static void processFunc( const Conn& c, ProcInfo p );

		/**
		 * This function receives the ranks of neurons it will receive Spikes from
		 */
		static void recvRank( const Conn& c, int rank );

		void innerProcessFunc( Element* e, ProcInfo info );
		unsigned int updateNumSynapse( const Element* e );
		void innerReinitFunc( Element* e, ProcInfo info );

	private:

		/**
		 * Stores the ranks from which a spike will be sent
		 */
	        vector< int > recvRank_;

};
#endif // _ParSynChan_h
