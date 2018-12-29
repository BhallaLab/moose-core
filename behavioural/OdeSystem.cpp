/***
 *    Description:  Ode System.
 *
 *        Created:  2018-12-28

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */

#include "OdeSystem.h"
#include "../utility/print_function.hpp"
#include "../utility/strutil.h"

namespace moose {

OdeSystem::OdeSystem()
{
}

OdeSystem::~OdeSystem()
{
}

void OdeSystem::setEquations(const vector<string>& eqs)
{
    eqs_ = eqs;
}

vector<string> OdeSystem::getEquations(void) const
{
    return eqs_;
}



bool OdeSystem::isValid(void)
{
    return isValid_;
}

} // namespace moose

