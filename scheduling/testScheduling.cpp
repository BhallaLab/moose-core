/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Message.h"
#include "Tick.h"
#include "TickPtr.h"
#include "Clock.h"

/**
 * Check that the ticks are set up properly, created and destroyed as
 * needed, and are sorted when dts are assigned
 */
void setupTicks()
{
	const Cinfo* tc = Tick::initCinfo();
	Id clock = Clock::initCinfo()->create( "tclock", 1 );
	Element* clocke = clock();
	Eref clocker = clock.eref();
	FieldElement< Tick, Clock, &Clock::getTick > tick( tc, clocke, 
		&Clock::getNumTicks, &Clock::setNumTicks );
	// TickElement tick( tc, clocke );
	assert( tick.numData() == 0 );

	// bool ret = SetGet2< double, unsigned int >::set( );
	cout << "." << flush;
	delete clocke;
}

void testScheduling( )
{
	setupTicks();
}
