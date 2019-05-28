/***
 *       Filename:  Parser.h
 *
 *    Description:  Parser class. Similar API as muParser.
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL3
 */

#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <memory>
#include <exception>
#include <map>
#include <iostream>

#include "../external/exprtk/exprtk.hpp"

using namespace std;

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
typedef double value_type;
typedef map<string, value_type> varmap_type;
} // namespace Parser

class MooseParser
{
public:
    MooseParser();
    ~MooseParser();

    // Parser should never be copied. Else multithreaded programs MAY
    // produce unexptected results. I.e. FuncTerm must have its own parser.
    MooseParser& operator=(const moose::MooseParser&) = delete;

    // Other functions.
    void Reinit();

    /*-----------------------------------------------------------------------------
     *  Set/Get
     *-----------------------------------------------------------------------------*/
    Parser::symbol_table_t  GetSymbolTable( ) const;
    Parser::expression_t    GetExpression( ) const;

    void SetSymbolTable( Parser::symbol_table_t tab );

    /*-----------------------------------------------------------------------------
     *  User interface.
     *-----------------------------------------------------------------------------*/
    bool DefineVar( const string varName, double* v );

    void DefineConst( const string& cname, const double val );

    void DefineFun1( const string& funcName, double (&func)(double) );

    bool SetExpr( const string& expr );

    bool CompileExpr();

    // Reformat the expression to meet TkExpr.
    string Reformat( const string user_expr );

    void SetVariableMap( const map<string, double*> map );

    static void findAllVarsRegex( const string& expr, set<string>& vars, const string& start );
    static void findAllVars( const string& expr, set<string>& vars, const char p);

    static void findXsYs(const string& expr, set<string>& xs, set<string>& ys);

    double Eval( ) const;

    Parser::varmap_type GetVar() const;

    double Diff( const double a, const double b) const;

    Parser::varmap_type GetConst( ) const;
    Parser::varmap_type GetUsedVar( );

    void ClearVariables( );
    void ClearAll( );

    const string GetExpr( ) const;

    /*-----------------------------------------------------------------------------
     *  User defined function of parser.
     *-----------------------------------------------------------------------------*/
    static double Rand( );
    static double SRand( double seed );
    static double Rand2( double a, double b );
    static double SRand2( double a, double b, double seed );
    static double Fmod( double a, double b );


private:
    /* data */
    string expr_;
    double value=0.0;
    Parser::varmap_type var_map_;
    Parser::varmap_type const_map_;
    Parser::varmap_type used_vars_;

    /* Map to variable names and pointer to their values. */
    map<string, double*> map_;

    /* Parser related */
    // unique_ptr<Parser::symbol_table_t> symbol_table_;   [> symbol table <]
    Parser::symbol_table_t symbol_table_;
    Parser::expression_t   expression_;     /* expression type */
    Parser::parser_t       parser_;          /* parser */
    size_t num_user_defined_funcs_ = 0;

    bool symbol_tables_registered_;
};

} // namespace moose.

#endif /* end of include guard: PARSER_H */
