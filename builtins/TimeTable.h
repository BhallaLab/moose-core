/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class TimeTable
{
	public:
		TimeTable();
		~TimeTable();

		/* Functions to set and get TimeTable fields */
		static void setFilename(
				const Conn* c, 
				string filename );
		static string getFilename( Eref e );
		
		static void setMethod(
				const Conn* c, 
				int method );
		static int getMethod( Eref e );
		
		static void setTableVector(
				const Conn* c, 
				vector< double > table );
		static vector< double > getTableVector( Eref e );
		
		static void setTableSize(
				const Conn* c, 
				unsigned int val );
		static unsigned int getTableSize( Eref e );
		
		static void setTable(
				const Conn* c, 
				double val, 
				const unsigned int& i );
		static double getTable(
				Eref e,
				const unsigned int& i );
		
		static double getState( Eref e );
		
		/* Dest functions */
		/**
		 * The process function called by scheduler on every tick
		 */
		static void processFunc(
				const Conn* c,
				ProcInfo info );
		
		/**
		 * The reinit function called by scheduler for the reset 
		 */
		static void reinitFunc(
				const Conn* c,
				ProcInfo info );
	
	private:
		/*
		 * Object (non-static) functions
		 */
		void localSetFilename(
				string filename );
		void localSetTable(
				double value,
				unsigned int index );
		double localGetTable(
				unsigned int index ) const;
		void processFuncLocal(
				Eref e,
				ProcInfo info );
		void reinitFuncLocal( );
		
		
		/*
		 * Fields
		 */
		string filename_;
		
		/* The table with (spike)times */
		vector < double > timeTable_;
		
		double state_;
		
		/* Current position within the table */
		unsigned int curPos_;
		
		/* How to fill the timetable, 
		   currently only 4 = reading from ASCII file is supported */
		int method_;

};
