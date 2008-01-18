/*******************************************************************
 * File:            NumUtil.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-02 11:47:21
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NUMUTIL_H
#define _NUMUTIL_H

#include <float.h>
const int WORD_LENGTH = 32; // number of bits in a word - check for portability
const double LN2 = 0.69314718055994528622676;
const unsigned long LN2BYTES = 0xB1721814;
const double NATURAL_E = 2.718281828459045;

//extern const double getMachineEpsilon();
//extern const double EPSILON;
#ifndef M_PI
#define M_PI 3.14159265358979323846             
#endif


bool isEqual(float x, float y, float epsilon = FLT_EPSILON);
bool isEqual(double x, double y, double epsilon = DBL_EPSILON);
bool isEqual(long double x, long double y, long double epsilon = LDBL_EPSILON);
// round, isinf and isnan are not defined in VC++ or Borland C++
#if defined(__TURBOC__) || defined(__BORLANDC__) || defined(_MSC_VER)
#define isinf(param) !_finite(param)
#define isnan(param) _isnan(param)
#define round(param) floor(param+0.5)
#endif
#endif
