/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include "header.h"

extern void testCompartment(); // Defined in Compartment.cpp
extern void testHHChannel(); // Defined in HHChannel.cpp
extern void testCaConc(); // Defined in CaConc.cpp
extern void testNernst(); // Defined in Nernst.cpp
extern void testSpikeGen(); // Defined in SpikeGen.cpp
extern void testSynChan(); // Defined in SynChan.cpp
extern void testBioScan(); // Defined in BioScan.cpp

void testBiophysics()
{
	testCompartment();
	testHHChannel();
	testCaConc();
	testNernst();
	testSpikeGen();
	testSynChan();
	testBioScan();
}

#endif
