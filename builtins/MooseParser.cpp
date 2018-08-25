/***
 *    Description:  Moose Parser class.
 *
 *        Created:  2018-08-25

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */
#include "MooseParser.h"

#include "../external/tinyexpr/tinyexpr.h"
#include "../utility/print_function.hpp"

namespace moose {

MooseParser::MooseParser() {;}
MooseParser::~MooseParser() {;}

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

}

void MooseParser::SetExpr( const string& expr )
{

    MOOSE_DEBUG( "Setting expression " << expr );
    expr_ = expr;
}

moose::Parser::value_type MooseParser::Eval( ) const
{
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
