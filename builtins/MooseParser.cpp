/***
 *    Description:  Moose Parser class, wraps exprtk.
 *
 *        Created:  2018-08-25

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 */

#include <vector>
#include <cassert>
#include <regex>

#include "../utility/testing_macros.hpp"
#include "../utility/print_function.hpp"
#include "../utility/strutil.h"
#include "../basecode/global.h"
#include "../builtins/Variable.h"
#include "MooseParser.h"

using namespace std;

namespace moose
{

MooseParser::MooseParser()
{
    Parser::symbol_table_t symbol_table;

    // And add user defined functions.
    symbol_table.add_function( "ln", MooseParser::Ln );
    symbol_table.add_function( "rand", MooseParser::Rand ); // between 0 and 1
    symbol_table.add_function( "rnd", MooseParser::Rand );  // between 0 and 1

    symbol_table.add_function( "srand", MooseParser::SRand );
    symbol_table.add_function( "rand2", MooseParser::Rand2 );
    symbol_table.add_function( "srand2", MooseParser::SRand2 );
    symbol_table.add_function( "fmod", MooseParser::Fmod );

    expression_.register_symbol_table(symbol_table);
}

MooseParser::~MooseParser()
{
}

/*-----------------------------------------------------------------------------
 *  User defined function here.
 *-----------------------------------------------------------------------------*/
double MooseParser::Ln( double v )
{
    return std::log(v);
}

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
Parser::symbol_table_t& MooseParser::GetSymbolTable()
{
    return expression_.get_symbol_table();
}

const Parser::symbol_table_t& MooseParser::GetSymbolTable() const
{
    return expression_.get_symbol_table();
}


double MooseParser::GetVarValue(const string& name) const
{
    return GetSymbolTable().get_variable(name)->value();
}

void MooseParser::findAllVars( const string& expr, set<string>& vars, const string& pattern)
{
    const regex xpat(pattern);
    smatch sm;
    string temp(expr);
    while(regex_search(temp, sm, xpat))
    {
        vars.insert(sm.str());
        temp = sm.suffix();
    }
}

/*-----------------------------------------------------------------------------
 *  Other function.
 *-----------------------------------------------------------------------------*/
bool MooseParser::DefineVar( const string varName, double* const val)
{
    // Use in copy assignment.
    if( GetSymbolTable().is_variable(varName))
        GetSymbolTable().remove_variable(varName, false);
    return GetSymbolTable().add_variable(varName, *val, false);
}

void MooseParser::DefineConst( const string& constName, const double value )
{
    const_map_[constName] = value;
    GetSymbolTable().add_constant(constName, value);
}

void MooseParser::DefineFun1( const string& funcName, double (&func)(double) )
{
    // Add a function. This function currently handles only one argument
    // function.
    num_user_defined_funcs_ += 1;
    GetSymbolTable().add_function( funcName, func );
}


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  EXPRTK does not have && and || but have 'and' and 'or' symbol.
 * Replace && with 'and' and '||' with 'or'.
 *
 * @Param user_expr
 *
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
string MooseParser::Reformat( const string user_expr )
{
    string expr( user_expr );

    // Replate || with 'or'
    moose::str_replace_all( expr, "||", " or " );
    // Replace && with 'and'
    moose::str_replace_all( expr, "&&", " and " );

    // Trickt business: Replace ! with not but do not change !=
    moose::str_replace_all( expr, "!=", "@@@" ); // placeholder
    moose::str_replace_all( expr, "!", " not " );
    moose::str_replace_all( expr, "@@@", "!=" ); // change back @@@ to !=

    return expr;
}


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Find all x\d+ and y\d+ in the experssion.
 *
 * @Param expr
 * @Param vars
 */
/* ----------------------------------------------------------------------------*/
void MooseParser::findXsYs( const string& expr, set<string>& xs, set<string>& ys )
{
    findAllVars( expr, xs, "x\\d+");
    findAllVars( expr, ys, "y\\d+" );
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Set expression on parser.
 *
 * @Param user_expr
 *
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
bool MooseParser::SetExpr( const string& user_expr )
{
    ASSERT_FALSE( user_expr.empty(), "Empty expression" );
    expr_ = Reformat(user_expr);
    return CompileExpr();
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Compile a given expression.
 *
 * @Returns Return true if successful, throws exception if compilation fails.
 * Exception includes a detailed diagnostic.
 */
/* ----------------------------------------------------------------------------*/
bool MooseParser::CompileExpr()
{
    // User should make sure that symbol table has been setup. Do not raise
    // exception here. User can set expression again.
    // GCC specific
    ASSERT_FALSE(expr_.empty(), __func__ << ": Empty expression not allowed here");

    Parser::parser_t  parser;
    auto res = parser.compile(expr_, expression_);
    if(! res)
    {
        std::stringstream ss;
        ss << "Failed to parse '" << expr_ << "' :" << endl;
        for (std::size_t i = 0; i < parser.error_count(); ++i)
        {
            Parser::error_t error = parser.get_error(i);
            ss << "Error[" << i << "] Position: " << error.token.position
               << " Type: [" << exprtk::parser_error::to_str(error.mode)
               << "] Msg: " << error.diagnostic << endl;

            // map is
            auto symbTable = GetSymbolTable();
            vector<std::pair<string, double>> vars;
            auto n = symbTable.get_variable_list(vars);
            ss << "More Information:\nTotal variables " << n << ".";
            for (auto i : vars)
                ss << "\t" << i.first << "=" << i.second << " " << symbTable.get_variable(i.first)->ref();
            ss << endl;
        }
        // Throw the error, this is handled in callee.
        throw moose::Parser::exception_type(ss.str());
    }
    return res;
}

double MooseParser::Derivative(const string& name) const
{
    return exprtk::derivative(expression_, name);
}

double MooseParser::Eval(bool check) const
{
    if( expr_.empty())
        return 0.0;
    double v = expression_.value();
    if(check)
    {
        if(! std::isfinite(v)) 
        {
            stringstream ss;
            auto symbTable = GetSymbolTable();
            vector<std::pair<string, double>> vars;
            auto n = symbTable.get_variable_list(vars);
            ss << "| Vars (" << n << ") ";
            for (auto i : vars)
                ss << i.first << "=" << i.second << ", ";
            cerr << "Warn: Expr: " << expr_ << "-> " << v 
                << endl << ss.str() << endl;
        }
    }
    return v;
}


double MooseParser::Diff( const double a, const double b ) const
{
    return a-b;
}

Parser::varmap_type MooseParser::GetConst( ) const
{
    return const_map_;
}

void MooseParser::ClearVariables( )
{
    GetSymbolTable().clear_variables();
}

void MooseParser::ClearAll( )
{
    ClearVariables();
}

void MooseParser::Reset( )
{
    expression_.release();
}

const string MooseParser::GetExpr( ) const
{
    return expr_;
}

void MooseParser::LinkVariables(vector<Variable*>& xs, vector<double*>& ys, double* t)
{
    for(size_t i = 0; i < xs.size(); i++)
        DefineVar('x'+to_string(i), xs[i]->ref());

    for (size_t i = 0; i < ys.size(); i++) 
        DefineVar('y'+to_string(i), ys[i]);

    DefineVar("t", t);
}

void MooseParser::LinkVariables(vector<shared_ptr<Variable>>& xs, vector<shared_ptr<double>>& ys, double* t)
{
    for(size_t i = 0; i < xs.size(); i++)
        DefineVar('x'+to_string(i), xs[i]->ref());

    for (size_t i = 0; i < ys.size(); i++) 
        DefineVar('y'+to_string(i), ys[i].get());

    DefineVar("t", t);
}



} // namespace moose.
