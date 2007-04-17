/*******************************************************************
 * File:            PyMooseUtil.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-03-23 16:03:01
 ********************************************************************/
#ifndef _PYMOOSE_UTIL_CPP
#define _PYMOOSE_UTIL_CPP
#include <iostream>
#include <sstream>
#include <string>
#include "PyMooseUtil.h"
bool isEqual(double val1, double val2, double epsilon)
{
    bool result = false;
    double diff;
    assert(epsilon > 0);
    
    if ((val1 == 0) && (val2 == 0))
    {
        return true;
    }
    
    diff = (val1>val2) ? (val1-val2):(val2-val1);
    if (val1 == 0)
    {
        if (val2 > 0)
        {
            result = (val2 < epsilon);
        }
        else 
        {
            result = (val2 > epsilon);
        }
    } else if ( val1 > 0 )
    {
        result = diff/val1 < epsilon;
    }
    else
    {
        result = (-diff/val1) < epsilon;
    }
    return result;
}

    
    

#endif // _PYMOOSE_UTIL_CPP
