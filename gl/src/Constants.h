/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <limits>

#ifndef CONSTANTS_H
#define CONSTANTS_H

const double SIZE_EPSILON = 1e-8; // floating-point (FP) epsilon for 
                                  // ... minimum compartment size

//const double FP_EPSILON = 1e-8;   // FP epsilon for comparison
const double FP_EPSILON = std::numeric_limits<double>::epsilon();

const int WINDOW_OFFSET_X = 50;
const int WINDOW_OFFSET_Y = 50;
const int WINDOW_WIDTH = 600;
const int WINDOW_HEIGHT = 600;

const char SYNCMODE_ACKCHAR = '*';

const double DEFAULT_INCREMENT_ANGLE = 10;

const double VALUE_MIN_DEFAULT = 0.0;
const double VALUE_MAX_DEFAULT = 1.0;
const unsigned int POINT_PARTICLE_DIAMETER = 1;

enum MsgType
{
	RESET,
	PROCESS_COLORS,
	PROCESS_COLORS_SYNC,
	PROCESS_PARTICLES,
	PROCESS_PARTICLES_SYNC,
	PROCESS_SMOLDYN_SHAPES,
	DISCONNECT
};

#endif // CONSTANTS_H
