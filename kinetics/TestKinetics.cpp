/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include <math.h>
#include "header.h"
#include "moose.h"
#include "../element/Neutral.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"

extern void testMolecule(); // Defined in Molecule.cpp
extern void testEnzyme(); // Defined in Enzyme.cpp
extern void testMathFunc(); //Defined in MathFunc.cpp

void testKinetics()
{
	testMolecule();
	testEnzyme();
	testMathFunc();
}

#endif
