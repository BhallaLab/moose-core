/***
 *    Description:  Moose Parser class.
 *
 *        Created:  2018-08-25

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */

#include <vector>
#include <cassert>

#include "../utility/testing_macros.hpp"
#include "../utility/print_function.hpp"
#include "../utility/strutil.h"
#include "../basecode/global.h"
#include "MooseParser.h"
// #define DEBUG_HERE

using namespace std;

namespace moose
{

MooseParser::MooseParser() : symbol_tables_registered_(false)
{
    symbol_table_ = unique_ptr<Parser::symbol_table_t>(new Parser::symbol_table_t());

    // And add user defined functions.
    symbol_table_->add_function( "rand", MooseParser::Rand );
    symbol_table_->add_function( "srand", MooseParser::SRand );
    symbol_table_->add_function( "rand2", MooseParser::Rand2 );
    symbol_table_->add_function( "srand2", MooseParser::SRand2 );
    symbol_table_->add_function( "fmod", MooseParser::Fmod );
    expression_.register_symbol_table(*symbol_table_.get());
}

MooseParser::~MooseParser()
{
}

/*-----------------------------------------------------------------------------
 *  User defined function here.
 *-----------------------------------------------------------------------------*/
double MooseParser::Rand( )
{
    return moose::mtrand();
}

double MooseParser::SRand( double seed = -1 )
{
    if( seed >= 0 )
        moose::mtseed( (size_t) seed );
    return moose::mtrand();
}

double MooseParser::Rand2( double a, double b )
{
    return moose::mtrand( a, b );
}

double MooseParser::SRand2( double a, double b, double seed = -1 )
{
    if( seed >= 0 )
        moose::mtseed( (size_t) seed );
    return moose::mtrand( a, b );
}

double MooseParser::Fmod( double a, double b )
{
    return fmod(a, b);
}


/*-----------------------------------------------------------------------------
 *  Get/Set
 *-----------------------------------------------------------------------------*/
Parser::symbol_table_t MooseParser::GetSymbolTable( ) const
{
    return expression_.get_symbol_table();
}

Parser::expression_t MooseParser::GetExpression( ) const
{
    return expression_;
}

/*-----------------------------------------------------------------------------
 *  Other function.
 *-----------------------------------------------------------------------------*/
void MooseParser::DefineVar( const string& varName, double& val)
{
    // If this variable alreay exists, then delete the previous instance and
    // create the new variable.
    if(0 == symbol_table_->variable_ref(varName))
    {
#ifdef DEBUG_HERE
        MOOSE_DEBUG( "++ Adding var " << varName << "=" << val << "(" << &val << ")");
#endif
        symbol_table_->add_variable(varName, val);
        return;
    }
    //throw runtime_error("Variable " + varName + " already exists parser's table.");
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Reinit the parser. Reassign symbol table, recompile the
 * expression.
 */
/* ----------------------------------------------------------------------------*/
void MooseParser::Reinit( )
{
    ClearVariables();
    expression_.register_symbol_table(*symbol_table_.get());
}

void MooseParser::DefineConst( const string& constName, const double value )
{
    const_map_[constName] = value;
}

void MooseParser::DefineFun1( const string& funcName, double (&func)(double) )
{
    // Add a function. This function currently handles only one argument
    // function.
    num_user_defined_funcs_ += 1;
    symbol_table_->add_function( funcName, func );
}

bool MooseParser::IsConstantExpr( const string& expr )
{
    vector<string> xs;
    vector<string> ys;
    findXsYs( expr, xs, ys );
    if( 0 < (xs.size() + ys.size()) )
        return false;
    return true;
}

void MooseParser::findAllVars( const string& expr, vector<string>& vars, char start )
{
    int startVar=-1;
    string temp = expr + "!"; // To make sure we compare the last element as well.
    for( size_t i = 0; i < temp.size(); i++)
    {
        if( startVar == -1 && start == expr[i] )
        {
            startVar = i;
            continue;
        }
        if( startVar > -1 )
        {
            if( ! isdigit( expr[i] ) )
            {
                vars.push_back( expr.substr(startVar, i - startVar ) );
                startVar = -1;
            }
        }
    }
}

string MooseParser::Reformat( const string user_expr )
{
    string expr( user_expr );

    // Replate || with or
    moose::str_replace_all( expr, "||", " or " );
    moose::str_replace_all( expr, "&&", " and " );

    // replace ! with not but do not change !=
    moose::str_replace_all( expr, "!=", "@@@" ); // placeholder
    moose::str_replace_all( expr, "!", " not " );
    moose::str_replace_all( expr, "@@@", "!=" ); // change back @@@ to !=

    return expr;
}

void MooseParser::findXsYs( const string& expr, vector<string>& xs, vector<string>& ys )
{
    findAllVars( expr, xs, 'x' );
    findAllVars( expr, ys, 'y' );
}

bool MooseParser::SetExpr( const string& user_expr )
{
    expr_ = moose::trim(user_expr);
    expr_ = Reformat(expr_);
    if(expr_.empty())
        return false;
    return CompileExpr();
}

bool MooseParser::CompileExpr()
{
    // User should make sure that symbol table has been setup. Do not raise
    // exception here. User can set expression again.
    MOOSE_DEBUG( this << ": Compiling " << expr_ );

    if(expr_.empty())
        return false;

    if(! parser_.compile(expr_, expression_))
    {
        stringstream ss;
        for (std::size_t i = 0; i < parser_.error_count(); ++i)
        {
            Parser::error_t error = parser_.get_error(i);
            ss << "Error[" << i << "] Position: " << error.token.position
                 << " Type: [" << exprtk::parser_error::to_str(error.mode)
                 << "] Msg: " << error.diagnostic << endl;
        }
        throw runtime_error("Error in compilation: " + expr_ + "\n" + ss.str());
    }
    return true;
}

void MooseParser::SetVariableMap( const map<string, double*> m )
{
    map_.clear();
    for( auto &v : m )
    {
        map_[v.first] = v.second;
        symbol_table_->add_variable( v.first, *v.second );
    }
}

double MooseParser::Eval( ) const
{
#ifdef DEBUG_HERE
    // Print symbol table.
    vector<std::pair<string, double>> vars;
    auto symbTable = GetSymbolTable();
    auto n = symbTable.get_variable_list(vars);
    cout << "Eval(): Total variables " << n << ".";
    for (auto i : vars)
    {
        cout << "\t" << i.first << "=" << i.second << " " << &symbol_table_->get_variable(i.first)->ref();
    }
    cout << endl;
#endif
    if( expr_.empty())
        return 0.0;
    return expression_();
}

Parser::varmap_type MooseParser::GetVar() const
{
    return var_map_;
}


double MooseParser::Diff( const double a, const double b ) const
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

void MooseParser::ClearVariables( )
{
    symbol_table_->clear_variables();
}

void MooseParser::ClearAll( )
{
    const_map_.clear();
    var_map_.clear();
    ClearVariables();
}

const string MooseParser::GetExpr( ) const
{
    return expr_;
}

void MooseParser::SetVarFactory( double* (*fn)(const char*, void*), void *)
{
    MOOSE_WARN( "setVarFactory is not implemented." );
    throw;
}


} // namespace moose.
