/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This class is an example Moose class. This class explains the
 * functions a Moose class needs to export/define to communicate with
 * the Moose messaging architecture. Every Moose class is needs to define
 * certain specific functions for the Moose messaging architecture.
 * Moose data members are exposed using functions.
 *
 * A Moose class needs to expose it's data member using get / set functions
 * for them. These functions should be defined with a specific syntax.
 * See get/set functions for data members: total_, baseline_, n_.
 *
 * Moose messaging architecture also expects certain static functions to be defined:
 * A "process" function which is called by the scheduler on every tick.
 * This function implements the functionality to be executed on every clock
 * tick. Normally a non-static member function "innerProcess" is defined which is
 * called from "process" functions so data members are directly accessible. In this
 * class the "innerProcess" function calculates the average value on each tick.
 *
 * A "reinit" function which is called by the scheduler for every "reset" command in
 * the script file. This function is expected to reset all the member variables to their
 * default values.
 *
 * An "input" function which accepts incoming data from other objects. This function
 * accepts data from other interconnected Moose class objects.
 *
 * The function explained above should be exposed to Moose in the following way:
 * Define a static Cinfo class object ("averageCinfo"). This object contains the
 * following information of the above defined functions:
 * A static "SchedInfo" object which contains the "process" and "reinit" function
 * pointers required by the scheduler.
 * A static array of the "Finfos". This array contains the following:
 * "ValueFinfo" objects with function pointer of get/set function for member variables
 * "SrcInfo" objects used to send out data to other connected objects on every tick.
 * "DestInfo" objects with function pointer of function to be called on receiving a
 * data from other objects.
 *
 * This class calculates the average value on every tick and send out the calculated average
 * value to other connected objects. This class is a sample class to explain the procedure
 * required for a Moose developer to add new classes.
 */
class TimeTable
{
	public:
		TimeTable();
                ~TimeTable();

		static void process( const Conn* c, ProcInfo p );

                /* Set max time for the time table */
                static void setMaxTime( const Conn* c, double time);


                /* Return max time for the time table */
                static double getMaxTime( Eref e );


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

                /* Max time */
                double maxTime_;

                double state_;

                /* Current position within the table */
                unsigned int curPos_;
                
                /* How to fill the timetable, 
                   currently only 4 = reading from ASCII file is supported */
                int method_;

};
