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

const int WORD_LENGTH = 32; // number of bits in a word - check for portability
const double LN2 = 0.69314718055994528622676;
const unsigned long LN2BYTES = 0xB1721814;
const double NATURAL_E = 2.718281828459045;

extern const double getMachineEpsilon();

#endif
