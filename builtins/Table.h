/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TABLE_H
#define _TABLE_H

/**
 * Unlike the old GENESIS table, the options to this class are mostly done
 * through the messaging.
 * If there is a message coming in, it fills the vector.
 * If there is a message going out, it emits the vector.
 * If the outgoing message is the loop one, the output data loops.
 * If the incoming message is a spike message, it thresholds
 * 	and stores the spike time.
 */
class Table: public Data
{
	public: 
		Table();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setThreshold( double v );
		double getThreshold() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void reinit();

		void input( double v );
		void spike( double v );
		void xplot( string file, string plotname );

		//////////////////////////////////////////////////////////////////
		// Lookup funcs for table
		//////////////////////////////////////////////////////////////////
		double* lookupVec( unsigned int index );
		void setVecSize( unsigned int num );
		unsigned int getVecSize( ) const;

		// bool isInside( double x, double y, double z );

		static const Cinfo* initCinfo();
	private:
		double threshold_;
		double lastTime_;
		unsigned int outputIndex_;
		vector< double > vec_;
};

#endif	// _TABLE_H
