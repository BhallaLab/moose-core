/*******************************************************************
 * File:            Normal.h
 * Description:      Generates random numbers with normal
 *                      distribution.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-30 11:22:51
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

#ifndef _NORMAL_H
#define _NORMAL_H
#include "Probability.h"
#define WORD_LENGTH 32 // number of bits in a word
#define LN2 0.69314718055994528622676
enum NormalGenerator
{
    ALIAS,
    BOXMUELLER
};

class Normal : public Probability
{
    
  public:
    Normal();
    Normal(NormalGenerator);
    Normal(double mean, double variance);
    
    double getMean();
    double getVariance();
    double getNextSample();
  private:
    double mean_;
    double variance_;
    double (*generator_)();
    
    static double BoxMueller();
    static double aliasMethod();
    static bool testAcceptance(unsigned long t, unsigned long v);    
};

    
#endif
