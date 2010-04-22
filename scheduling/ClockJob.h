/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CLOCKJOB_H
#define _CLOCKJOB_H

class ClockJob
{
	public:
		ClockJob();

		//////////////////////////////////////////////////////////
		//  Field assignment functions
		//////////////////////////////////////////////////////////
		static void setRunTime( const Conn* c, double v );
		static double getRunTime( Eref e );
		static double getCurrentTime( Eref e );
		static void setNsteps( const Conn* c, int v );
		static int getNsteps( Eref e );
		static int getCurrentStep( Eref e );
                static int getAutoschedule( Eref e);
                static void setAutoschedule(const Conn* c, int v);
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		static void receiveNextTime( const Conn*, double nextTime );
		static void startFunc( const Conn* c, double runTime );
		void startFuncLocal( Eref e, double runTime );
		static void stepFunc( const Conn* c, int nsteps );
		static void handleStopCallback( const Conn* c, int flag );
		static void handleRunningCallback( const Conn* c, bool isRunning );
		/**
		 * ReinitClock is used to reinit the state of the scheduling system.
		 * This does not send out reinit calls to objects connected to ticks.
		 */
		static void reinitClockFunc( const Conn* c );
		void reinitClockFuncLocal( Eref e );
		static void reinitFunc( const Conn* c );
		void reinitFuncLocal( Eref e );
		static void reschedFunc( const Conn* c );
		void reschedFuncLocal( Eref e );
		static void dtFunc( const Conn* c, double dt );
		void dtFuncLocal( Eref e, double dt );
    
		//////////////////////////////////////////////////////////
		//  Utility functions
		//////////////////////////////////////////////////////////
		void clearMessages( Eref e );
		// void buildMessages( Element* last, Element* e );
		//////////////////////////////////////////////////////////
		//  static Globals
		//////////////////////////////////////////////////////////
		static const int emptyCallback;
		static const int doReschedCallback;

	private:
		double runTime_;
		double currentTime_;
		double nextTime_;
		int nSteps_;
		int currentStep_;
		double dt_;
		bool isRunning_;
		ProcInfoBase info_;
		int callback_;
                int autoschedule_;
};

#endif // _CLOCKJOB_H
