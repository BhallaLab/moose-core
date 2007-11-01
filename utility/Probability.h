/*******************************************************************
 * File:            Probability.h
 * Description:      This is base class for various probability
 *                   distribution generator classes.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-28 13:30:41
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

#ifndef _PROBABILITY_H
#define _PROBABILITY_H

const int WORD_LENGTH = 32; // number of bits in a word
const double LN2 = 0.69314718055994528622676;
const unsigned long LN2BYTES = 0xB1721814;

/**
   Base class for implementing various probability distributions.
 */
class Probability
{
  public:
    virtual ~Probability(){};
    
    virtual double getMean() const =0;
    virtual double getVariance()const =0;
    virtual double getNextSample()const =0;
    
  private:
//     long double mean_; // TODO : do we really need this?
//     long double variance_;// TODO : do we really need this?    
};

    
#endif
