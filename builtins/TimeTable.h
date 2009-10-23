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

		static void process( const Conn* c, ProcInfo p );

		/* Functions to set and get TimeTable */
		static void setTableVector( const Conn* c, 
									vector< double > table );

		static vector< double > getTableVector( Eref e );

		void localSetTableVector( const vector< double >& timeTable );


		/* Functions to get/set values in the time table */
		static void setTable(const Conn* c, 
							 double val, 
							 const unsigned int& i );

		static double getTable(Eref e,const unsigned int& i );

		void setTableValue( double value, unsigned int index );

		double getTableValue( unsigned int index ) const;

		/* Return the table size (there is no set function) */
		static unsigned int getTableSize( Eref e);

		double getState();


		/* Load data to table from file */
		static void load( const Conn* c, 
						  string fName,
						  unsigned int skipLines );

		void innerLoad( const string& fName, unsigned int skipLines );

		/**
		 * The process function called by scheduler on every tick
		 */
		static void processFunc( const Conn* c, ProcInfo info );
		/**
		 * The non-static function called by "processFunc" 
		 */
		void processFuncLocal( Eref e, ProcInfo info );

		/**
		 * The reinit function called by scheduler for the reset 
		 */
		static void reinitFunc( const Conn* c, ProcInfo info );
		/**
		 * The non-static function called by the "reinitFunc" function.
		 */
		void reinitFuncLocal( );
	
	private:

		/* The table with (spike)times */
		vector < double > timeTable_;

		double state_;

		/* Current position within the table */
		unsigned int curPos_;

		/* How to fill the timetable, 
		   currently only 4 = reading from ASCII file is supported */
		int method_;

};
