/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
**             and Niraj Dudani and Johannes Hjorth, KTH, Stockholm
**
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MOOSE_MUSIC_H
#define _MOOSE_MUSIC_H

class Music 
{
public:
	
//////////////////////////////////////////////////////////////////
// Field access functions
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Message dest functions.
//////////////////////////////////////////////////////////////////
	static void processFunc( const Conn* c, ProcInfo p );
	static void reinitFunc( const Conn* c, ProcInfo p );
	static void reinitializeFunc( const Conn* c );
        static MPI::Intracomm setup( int& argc, char**& argv );

	static void finalizeFunc( const Conn* c );
	
	static void addPort(
		const Conn* c,
		string direction,
		string type,
		string name );


        static int getRank( Eref e );
        static int getSize( Eref e );
        static double getStopTime( Eref e );


protected:

private:

	void innerProcessFunc( const Conn* c, ProcInfo p );
        void innerFinalizeFunc( Eref e );
        void innerReinitFunc( Eref e, ProcInfo p );
        void innerReinitializeFunc( );

	void innerAddPort(
		Eref e,
		string direction,
		string type,
		string name );
	
	static MUSIC::Setup* setup_;
	static MUSIC::Runtime* runtime_;
	
	static double dt_;
        static double stopTime_;


};

#endif // MOOSE_MUSIC_H
