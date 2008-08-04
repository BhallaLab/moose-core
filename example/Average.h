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
class Average
{
	public:
		Average();


		static void process( const Conn* c, ProcInfo p );

		/**
		 * The input function called by other connected objects.
		 * v is value sent by the calling object
		 */
		static void input( const Conn* c, double v );

		/**
		 * The non-static function called by the input function.
		 */
		void inputLocal( double v );

		/**
		 * This function calculates average on the data members
		 */
		double mean( ) const;

		/**
		 * Retrieves the value of "total_" data member
		 */
		static double getTotal( Eref e );
		/**
		 * Set the value of "total_" data member
		 */
		static void setTotal( const Conn* c, double v );

		/**
		 * Retrieves the value of "baseline_" data member
		 */
		static double getBaseline( Eref e );
		/**
		 * Set the value of "baseline_" data member
		 */
		static void setBaseline( const Conn* c, double v );

		/**
		 * Retrieves the value of "n_" data member
		 */
		static unsigned int getN( Eref e );
		/**
		 * Set the value of "n_" data member
		 */
		static void setN( const Conn* c, unsigned int v );

		/**
		 * Calculates and returns the average value of data members
		 */
		static double getMean( Eref e );

		/**
		 * The process function called by scheduler on every tick
		 */
		static void processFunc( const Conn* c, ProcInfo info );
		/**
		 * The non-static function called by the "processFunc" function.
		 */
		void processFuncLocal( Eref e, ProcInfo info );

		/**
		 * The reinit function called by scheduler for the reset command
		 */
		static void reinitFunc( const Conn* c, ProcInfo info );
		/**
		 * The non-static function called by the "reinitFunc" function.
		 */
		void reinitFuncLocal( );
	private:

		/**
		 * Total value of all inputs received.
		 */
		double total_;

		/**
		 * Calculated average value.
		 */
		double baseline_;

		/**
		 * Total count of all inputs received.
		 */
		unsigned int n_;
};
