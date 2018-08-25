/***
 *    Description:  Moose Parser class.
 *
 *        Created:  2018-08-25

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */
#include <vector>
using namespace std;

#include "MooseParser.h"

#include "../external/tinyexpr/tinyexpr.h"
#include "../utility/print_function.hpp"

namespace moose
{

MooseParser::MooseParser() {;}

MooseParser::~MooseParser() 
{
    te_free( te_expr_ );
}

void MooseParser::DefineVar( const char* varName, moose::Parser::value_type* val)
{
    cout << "NA " << varName << " val: " << val << endl;

}

void MooseParser::DefineVar( const string& varName, moose::Parser::value_type& val)
{
    cout << "NA " << varName << " val: " << val << endl;
}

void MooseParser::DefineFun( const char* funcName, moose::Parser::value_type (&func)(moose::Parser::value_type) )
{
    cout << "NA: Defining func " << endl;
}

bool MooseParser::IsConstantExpr( const string& expr )
{
    vector<string> vars;
    findXsYs( expr, vars );
    if( vars.size() > 0 )
        return false;
    return true;
}

void MooseParser::findXsYs( const string& expr, vector<string>& vars )
{
    size_t startVar=0, endVar=0; 
    for( size_t i = 0; i < expr.size(); i++)
    {
        if( startVar == 0 && ('x' == expr[i] || 'y' == expr[i]) )
        {
            startVar = i;
            continue;
        }
        if( startVar > 0 )
        {
            if( ! isdigit( expr[i] ) )
            {
                vars.push_back( expr.substr(startVar, i - startVar ) );
                startVar = 0;
            }
        }
    }
}

void MooseParser::SetExpr( const string& expr )
{
    MOOSE_DEBUG( "Setting expression " << expr );
    if( map_.empty() )
    {
        MOOSE_WARN( "Parser does not know the value of x0, y0 etc." );
        return;
    }

    size_t i = 0;
    for (auto itr = map_.begin(); itr != map_.end(); itr++)
    {
        te_variable t;
        t.name = itr->first.c_str();
        t.address = itr->second;
        te_vars_.push_back( t );
    }

    te_expr_ = te_compile( expr.c_str(), &te_vars_[0], map_.size(), err_ );
    if( te_expr_ == NULL )
    {
        MOOSE_WARN( "Failed to compile expression: " << expr << " . Error at " << err_ );
        return;
    }
}

moose::Parser::value_type MooseParser::Eval( ) const
{
    if( te_expr_ )
        return te_eval( te_expr_ );
    else
        MOOSE_WARN( "could not evalualte." );

    return 0.0;
}

Parser::varmap_type MooseParser::GetVar() const
{
    return var_map_;
}


void MooseParser::DefineConst( const string& constName, moose::Parser::value_type& value )
{
    MOOSE_DEBUG( "Adding constant " << constName << " with value " << value );
    const_map_[constName] = value;
}

void MooseParser::DefineConst( const char* constName, const moose::Parser::value_type& value )
{
    const_map_[constName] = value;
}

moose::Parser::value_type MooseParser::Diff( const moose::Parser::value_type a, const moose::Parser::value_type b) const
{
    return a-b;
}

Parser::varmap_type MooseParser::GetConst( ) const
{
    return const_map_;
}

Parser::varmap_type MooseParser::GetUsedVar( )
{
    return used_vars_;
}

void MooseParser::ClearVar( )
{
    const_map_.clear();
    var_map_.clear();
}

const string MooseParser::GetExpr( ) const
{
    return expr_;
}

void MooseParser::SetVarFactory( double* (*fn)(const char*, void*), void *)
{
    MOOSE_DEBUG( "setVarFactory is not implemented." );
}

} // namespace moose.
