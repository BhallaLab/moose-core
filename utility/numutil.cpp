/*******************************************************************
 * File:            numutil.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray dot subhasis at gmail dot com
 * Created:         2007-11-02 11:46:40
 ********************************************************************/
/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifndef _NUMUTIL_CPP
#define _NUMUTIL_CPP
#include <math.h>	// Solaris CC compiler did not like <cmath>
bool almostEqual(float x, float y, float epsilon)
{
    if (x == 0.0 && y == 0.0) {
        return true;
    }
    
    if (fabs(x) > fabs(y)) {
        return fabs((x - y) / x) < epsilon;
    } else {
        return fabs((x - y) / y) < epsilon;
    }
}

bool almostEqual(double x, double y, double epsilon)
{
    if (x == 0.0 && y == 0.0){
        return true;
    }
    if (fabs(x) > fabs(y)){
        return fabs((x - y) / x) < epsilon;
    } else {
        return fabs((x - y) / y) < epsilon;
    }
}

bool almostEqual(long double x, long double y, long double epsilon)
{
    if (x == 0.0 && y == 0.0){
        return true;
    }
    if (fabs(x) > fabs(y)){
        return fabs((x - y) / x) < epsilon;
    } else {
        return fabs((x - y) / y) < epsilon;
    }
}

#endif
