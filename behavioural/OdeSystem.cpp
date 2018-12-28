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
#include "../external/muparser/include/muParser.h"
#include "../external/muparser/include/muParserDLL.h"

namespace moose {

enum VariableType { DERIVATION, NORMAL };

VariableType typeOfVariable( const string& var )
{
    if( var.find( "/dt" ) != std::string::npos )
        return DERIVATION;

    if( var.back() == '\'' )
        return DERIVATION;
    return NORMAL;
}

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

void OdeSystem::buildSystem(void* obj)
{
    LOG( moose::debug, "Building system from following equations:" );

    map<string, string> m;
    for( auto eq : eqs_ )
    {
        size_t loc = eq.find( '=' );
        if( loc == std::string::npos)
        {
            LOG( moose::warning, "Invalid equation: " << eq );
            continue;
        }

        auto lhs = moose::trim(eq.substr(0, loc));
        auto rhs = moose::trim(eq.substr(loc+1));
        if( typeOfVariable(lhs) == NORMAL )
            eqsMap_[lhs] = rhs;
        else
            odeMap_[lhs] = rhs;
    }

    mu::Parser p;
    // Replace occurance of variables which were stored in eqsMap_ by their
    // values.
    for( auto v : odeMap_ )
    {
        p.SetExpr(v.second);
        auto vars = p.GetUsedVar();
        for( auto vv : vars )
            if( eqsMap_.find(vv.first) != eqsMap_.end() )
                v.second = moose::replaceAll( v.second, vv.first, "(" + eqsMap_[vv.first] + ")" );
    }

    isValid_ = true;
}

bool OdeSystem::isValid(void)
{
    return isValid_;
}

} // namespace moose

