/***
 *    Description:  Parser class. Similar API as muParser.
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  See the LICENSE.md file.
 */

#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <memory>
#include <exception>
#include <map>
#include <iostream>

#define exprtk_enabled_debugging 0
#define exprtk_disable_comments 1
#include "../external/exprtk/exprtk.hpp"

using namespace std;

class Variable;

namespace moose
{
namespace Parser
{

// ExprTk types.
typedef exprtk::symbol_table<double> symbol_table_t;
typedef exprtk::expression<double>     expression_t;
typedef exprtk::parser<double>             parser_t;
typedef exprtk::parser_error::type error_t;

struct ParserException : public std::exception
{
    ParserException( const string msg ) : msg_(msg) { ; }

    string GetMsg()
    {
        return msg_;
    }

    string msg_;
};

typedef ParserException exception_type;
typedef map<string, double> varmap_type;
} // namespace Parser

class MooseParser
{
public:
    MooseParser();
    ~MooseParser();

    /*-----------------------------------------------------------------------------
     *  Set/Get
     *-----------------------------------------------------------------------------*/
    Parser::symbol_table_t& GetSymbolTable( ) const;

    void SetSymbolTable( Parser::symbol_table_t tab );

    /*-----------------------------------------------------------------------------
     *  User interface.
     *-----------------------------------------------------------------------------*/
    bool DefineVar( const string varName, double* const v );

    void DefineConst( const string& cname, const double val );

    void DefineFun1( const string& funcName, double (&func)(double) );

    bool SetExpr( const string& expr );

    bool CompileExpr();

    // Reformat the expression to meet TkExpr.
    string Reformat( const string user_expr );

    static void findAllVars( const string& expr, set<string>& vars, const string& start );
    static void findXsYs(const string& expr, set<string>& xs, set<string>& ys);

    void LinkVariables(vector<Variable*>& xs_, vector<double*>& ys_, double* t);

    double Eval( ) const;
    double Derivative(const string& name) const;

    double Diff( const double a, const double b) const;

    Parser::varmap_type GetConst( ) const;
    double GetVarValue(const string& name) const;


    void ClearVariables( );
    void ClearAll( );

    const string GetExpr( ) const;

    /*-----------------------------------------------------------------------------
     *  User defined function of parser.
     *-----------------------------------------------------------------------------*/
    static double Ln(double v);
    static double Rand( );
    static double SRand( double seed );
    static double Rand2( double a, double b );
    static double SRand2( double a, double b, double seed );
    static double Fmod( double a, double b );

public:
    /* data */
    string expr_;
    double value=0.0;

    Parser::varmap_type const_map_;

    /* Map to variable names and pointer to their values. */
    // map<string, double*> refs_;

    Parser::expression_t* expression_;     /* expression type */
    size_t num_user_defined_funcs_ = 0;

    bool symbol_tables_registered_;
};

} // namespace moose.

#endif /* end of include guard: PARSER_H */
