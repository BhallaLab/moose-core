/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This class is a parallel version of SpikeGen class.
 * This file is compiled only when the parallelizatin flag is enabled.
 * This class derives from SpikeGen, the spike generator class for serial Moose. 
 *
 * 
 * This class refers to the base class, SpikeGen, for all spike generating functionality. 
 * Parallel moose parser would require overriding of some of the base class functionality. 
 * Such functions would be overridden in this class. 
 *
 * 
 */


#ifndef _ParSpikeGen_h
#define _ParSpikeGen_h

class ParSpikeGen : public SpikeGen
{
	public:
		ParSpikeGen();
		~ParSpikeGen();

		/**
		 * This function receives the set of ranks it will send spikes to
		 */
		static void sendRank( const Conn& c, int rank );
	        void innerProcessFunc( const Conn& c, ProcInfo p );
        	static void processFunc( const Conn& c, ProcInfo p );

	private:
	
		/**
		 * Stores the ranks to which spike will be sent
		 */
		vector < int > sendRank_;
		vector < MPI_Request* > request_;

};
#endif // _ParSpikeGen_h
