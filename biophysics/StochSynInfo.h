/*******************************************************************
 * File:            StochSynInfo.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-29 12:00:25
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

#ifndef _STOCHSYNINFO_H
#define _STOCHSYNINFO_H
#include "SynInfo.h"

class StochSynInfo: public SynInfo
{
  public:
    double releaseP;
    bool hasReleased;    
};

    
#endif
