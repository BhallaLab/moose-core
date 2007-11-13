/*******************************************************************
 * File:            SynReceptor.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-13 14:04:05
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

#ifndef _SYNRECEPTOR_H
#define _SYNRECEPTOR_H
#include "header.h"

class SynReceptor
{
  public:
    static double getScale(const Element* e);
    static void setScale(const Conn& c, double scale);
    static void setReceivedCount(const Conn& c, double count);
    static void processFunc(const Conn&c, ProcInfo p);
    static void reinitFunc(const Conn&c, ProcInfo p);
  private:
    double scale_;
    double receivedCount_;
};

    
#endif
