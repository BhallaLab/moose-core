/***
 *    Description:  Behavioural decription of dyanmical system.
 *
 *        Created:  2018-12-29

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */

#include "BehaviouralSystem.h"
#include "../utility/print_function.hpp"

using namespace moose;

BehaviouralSystem::BehaviouralSystem ()
{
    ode_ = new OdeSystem();
} 

BehaviouralSystem::~BehaviouralSystem ()
{
    if(ode_)
        delete ode_;
} 


OdeSystem* BehaviouralSystem::getODE() const
{
    return ode_;
}

