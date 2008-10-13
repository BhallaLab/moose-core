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
	Music():
		setup_( 0 ),
		runtime_( 0 )
	{
		;
	}
	
//////////////////////////////////////////////////////////////////
// Field access functions
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Message dest functions.
//////////////////////////////////////////////////////////////////
	static void processFunc( const Conn* c, ProcInfo p );
	static void reinitFunc( const Conn* c, ProcInfo p );
        static void setupFunc( const Conn* c, MUSIC::setup* setup );

	static void finalizeFunc( const Conn* c );
	
	static void addPort(
		const Conn* c,
		string name,
		string direction,
		string type );

protected:

private:

	void innerProcessFunc( const Conn* c, ProcInfo p );
        void innerSetupFunc( Eref e, MUSIC::setup* setup );
        void innerFinalizeFunc( Eref e );
        void innerReinitFunc( Eref e, ProcInfo p );

	void innerAddPort(
		Eref e,
		string name,
		string direction,
		string type );
	
	MUSIC::setup* setup_;
	MUSIC::runtime* runtime_;
};

#endif // MOOSE_MUSIC_H
