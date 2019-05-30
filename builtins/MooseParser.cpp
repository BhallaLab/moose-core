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
    // symbol_table_ = unique_ptr<Parser::symbol_table_t>(new Parser::symbol_table_t());

    // And add user defined functions.
    symbol_table_.add_function( "rand", MooseParser::Rand );
    symbol_table_.add_function( "srand", MooseParser::SRand );
    symbol_table_.add_function( "rand2", MooseParser::Rand2 );
    symbol_table_.add_function( "srand2", MooseParser::SRand2 );
    symbol_table_.add_function( "fmod", MooseParser::Fmod );
    expression_.register_symbol_table(symbol_table_);
}

MooseParser::~MooseParser()
{
    // Parser will take care of it
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
bool MooseParser::DefineVar( const string varName, double* val)
{
    // Does not add duplicate variables.
    return symbol_table_.add_variable(varName, *val, false);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Reinit the parser. Reassign symbol table, recompile the
 * expression. Called from FuncTerm.
 */
/* ----------------------------------------------------------------------------*/
void MooseParser::Reinit( )
{
    ClearAll();
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

// Don't use it with gcc-4.8.x . It has a broken support for <regex>
void MooseParser::findAllVarsRegex( const string& expr, set<string>& vars, const string& pattern)
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

void MooseParser::findAllVars( const string& expr, set<string>& vars, const char p)
{
    size_t i = 0;
    while(i < expr.size()-1)
    {
        if( expr[i] == p && std::isdigit(expr[i+1]))
        {
            string v = expr.substr(i, 2);
            i += 2;
            while(std::isdigit(expr[i]))
            {
                v += expr[i];
                i += 1;
            }
            vars.insert(v);
        }
        else
            i += 1;
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

void MooseParser::findXsYs( const string& expr, set<string>& xs, set<string>& ys )
{
#if USE_REGEX
    findAllVarsRegex( expr, xs, "x\\d+");
    findAllVarsRegex( expr, ys, "y\\d+" );
#else
    findAllVars( expr, xs, 'x');
    findAllVars( expr, ys, 'y' );
#endif
}

bool MooseParser::SetExpr( const string& user_expr )
{
    expr_ = moose::trim(user_expr);
    expr_ = Reformat(expr_);
    if(expr_.empty())
        return false;
    return CompileExpr();
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Symbol table to string (for debugging purpose).
 *
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
string MooseParser::SymbolTable2String( )
{
    stringstream ss;
    // map is
    auto symbTable = GetSymbolTable();
    vector<std::pair<string, double>> vars;
    auto n = symbTable.get_variable_list(vars);
    ss << "More Information:\nTotal variables " << n << ".";
    for (auto i : vars)
        ss << "\t" << i.first << "=" << i.second << " " << &symbol_table_.get_variable(i.first)->ref();
    return ss.str();
}

bool MooseParser::CompileExpr()
{
    // User should make sure that symbol table has been setup. Do not raise
    // exception here. User can set expression again.
    // GCC specific
    if(expr_.empty())
        return false;

    Parser::parser_t       parser;
    if(! parser.compile(expr_, expression_))
    {
        stringstream ss;
        for (std::size_t i = 0; i < parser.error_count(); ++i)
        {
            Parser::error_t error = parser.get_error(i);
            ss << "Error[" << i << "] Position: " << error.token.position
                 << " Type: [" << exprtk::parser_error::to_str(error.mode)
                 << "] Msg: " << error.diagnostic << endl;

            ss << SymbolTable2String() << endl;
        }
        cerr <<  ss.str() << endl;
        return false;
    }
    return true;
}

void MooseParser::SetVariableMap( const map<string, double*> m )
{
    for( auto &v : m )
        symbol_table_.add_variable( v.first, *v.second );
}

double MooseParser::Eval( ) const
{
    if( expr_.empty())
        return 0.0;
    return expression_.value();
}

vector<string> MooseParser::GetVariables() const
{
    vector<string> vars;
    symbol_table_.get_variable_list(vars);
    return vars;
}

double& MooseParser::GetVar(const string& name) const
{
    return symbol_table_.get_variable(name)->ref();
}


double MooseParser::Diff( const double a, const double b ) const
{
    return a-b;
}

map<string, double> MooseParser::GetConst( ) const
{
    return const_map_;
}


void MooseParser::ClearVariables( )
{
    // Do not invalidate the reference.
    symbol_table_.clear_variables(false);
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
