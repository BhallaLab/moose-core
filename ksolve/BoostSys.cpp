/*
 * =====================================================================================
 *
 *       Filename:  BoostSystem.cpp
 *
 *    Description:  Ode system described boost library.
 *
 *        Created:  04/11/2016 10:58:34 AM
 *       Compiler:  g++
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#include "BoostSys.h"
#include <iostream>

BoostSys::BoostSys( std::string method )
{  
    method_ = method;
}


BoostSys::~BoostSys()
{  }


std::string BoostSys::getMethod( )
{
    return method_;
}
