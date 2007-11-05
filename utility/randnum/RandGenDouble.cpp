/*******************************************************************
 * File:            RandGenDouble.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-05 10:20:57
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

#ifndef _RANDGENDOUBLE_CPP
#define _RANDGENDOUBLE_CPP

double NormalRng::getSample( const Element* e)
{
    return static_cast<NormalRng*>( e->data())->sample_;
}

double NormalRng::getMean( const Element *e)
{
    return static_cast<NormalRng*>(e->data())->mean_;
}

double NormalRng::getVariance(const Element* e)
{
    return static_cast<NormalRng*>(e->data())->variance_;
}
#endif
