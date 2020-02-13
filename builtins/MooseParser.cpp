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
#include <regex>

#include "../utility/testing_macros.hpp"
#include "../utility/print_function.hpp"
#include "../utility/strutil.h"
#include "../basecode/global.h"
#include "MooseParser.h"

#define DEBUG_HERE

using namespace std;

namespace moose
{

MooseParser::MooseParser() : symbol_tables_registered_(false)
{
    // And add user defined functions.
    symbol_table_.add_function( "ln", MooseParser::Ln );
    symbol_table_.add_function( "rand", MooseParser::Rand ); // between 0 and 1
    symbol_table_.add_function( "rnd", MooseParser::Rand );  // between 0 and 1

    symbol_table_.add_function( "srand", MooseParser::SRand );
    symbol_table_.add_function( "rand2", MooseParser::Rand2 );
    symbol_table_.add_function( "srand2", MooseParser::SRand2 );
    symbol_table_.add_function( "fmod", MooseParser::Fmod );
    expression_.register_symbol_table(symbol_table_);
}

MooseParser::~MooseParser()
{
    symbol_table_.clear();
    expression_.release();
}

MooseParser& MooseParser::operator=(const moose::MooseParser& other) 
{
    return *this;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Copy data from other parser.
 *
 * @Param parser
 */
/* ----------------------------------------------------------------------------*/
void MooseParser::CopyData(const moose::MooseParser& other)
{
    for(auto v: other.refs_)
        DefineVar(v.first, v.second);
    SetExpr(other.expr_);
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
bool MooseParser::DefineVar( const string varName, double* const val)
{
    // Use in copy assignment.
    refs_[varName] = val;
    return symbol_table_.add_variable(varName, *val, false);
}

double MooseParser::GetVarValue(const string& name) const
{
    return symbol_table_.get_variable(name)->value();
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
}

void MooseParser::DefineConst( const string& constName, const double value )
{
    const_map_[constName] = value;
    symbol_table_.add_constant(constName, value);
}

void MooseParser::DefineFun1( const string& funcName, double (&func)(double) )
{
    // Add a function. This function currently handles only one argument
    // function.
    num_user_defined_funcs_ += 1;
    symbol_table_.add_function( funcName, func );
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
    cerr << "DEBUG: Found " << vars.size() << " expr: " << expr << endl;
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

void MooseParser::findXsYs( const string& expr, set<string>& xs, set<string>& ys )
{
    findAllVars( expr, xs, "x\\d+");
    findAllVars( expr, ys, "y\\d+" );
}

bool MooseParser::SetExpr( const string& user_expr )
{
    expr_ = moose::trim(user_expr);
    if(expr_.empty())
        return false;
    expr_ = Reformat(expr_);
    return CompileExpr();
}

bool MooseParser::CompileExpr()
{
    // User should make sure that symbol table has been setup. Do not raise
    // exception here. User can set expression again.
    // GCC specific
    if(expr_.empty())
        return false;

    Parser::parser_t       parser;          /* parser */
    if(! parser.compile(expr_, expression_))
    {
        stringstream ss;
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
                ss << "\t" << i.first << "=" << i.second << " " << &symbol_table_.get_variable(i.first)->ref();
            ss << endl;
        }
        cerr <<  ss.str() << endl;
        throw runtime_error(ss.str());
    }
    return true;
}

#if 0
void MooseParser::SetVariableMap( const map<string, double*> m )
{
    for( auto &v : m )
        symbol_table_.add_variable( v.first, *v.second );
}
#endif

double MooseParser::Derivative(const string& name) const
{
    return exprtk::derivative(expression_, name);
}

double MooseParser::Eval( ) const
{
    if( expr_.empty())
        return 0.0;
    return expression_();
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
    // Do not invalidate the reference.
    refs_.clear();
    symbol_table_.clear_variables(true);
}

void MooseParser::ClearAll( )
{
    const_map_.clear();
    ClearVariables();
}

const string MooseParser::GetExpr( ) const
{
    return expr_;
}


} // namespace moose.
