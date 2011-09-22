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
class Table
{
	public: 
		Table();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setThreshold( double v );
		double getThreshold() const;

		vector< double > getVec() const;
		void setVec( vector< double > val );

		double getOutputValue() const;

		unsigned int getOutputIndex() const;

		double getY( unsigned int index ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		void input( double v );
		void spike( double v );
		void xplot( string file, string plotname );
		void recvData( PrepackedBuffer pb );
		void loadXplot( string fname, string plotname );
		void loadCSV( 
			string fname, int startLine, int colNum, char separator );
		void compareXplot( string fname, string plotname, string op );
		void compareVec( vector< double > other, string op );

		//////////////////////////////////////////////////////////////////
		// Lookup funcs for table
		//////////////////////////////////////////////////////////////////
		double* lookupVec( unsigned int index );
		void setVecSize( unsigned int num );
		unsigned int getVecSize( ) const;

		static const Cinfo* initCinfo();
	private:
		double threshold_;
		double lastTime_;
		double output_;
		unsigned int outputIndex_;
		double input_;
		vector< double > vec_;
};

#endif	// _TABLE_H
