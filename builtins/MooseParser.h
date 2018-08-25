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

#include "../external/tinyexpr/tinyexpr.h"

#include <string>
#include <exception>
#include <map>
#include <iostream>

using namespace std;

namespace moose {
    namespace Parser {

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
    }

class MooseParser
{
    public:
        MooseParser();
        ~MooseParser();

        void DefineVar( const char* varName, moose::Parser::value_type* val);

        void DefineVar( const string& varName, moose::Parser::value_type& val);

        void DefineFun( const char* funcName, moose::Parser::value_type (&func)(moose::Parser::value_type) );

        /* --------------------------------------------------------------------------*/
        /**
         * @Synopsis Compile expression using tinyexpr library. Further calls
         * uses this compiled expression.
         *
         * @Param expr Expression. It can only contain following variables:
         *  t, x0, x1, ..., y0, y1, .., * c0, c1 etc.
         *
         * @Returns True if expression is compiled succcessfully, false
         * otherwise.
         */
        /* ----------------------------------------------------------------------------*/
        bool SetExpr( const string& expr );

        void SetVariableMap( const map<string, double*> map );

        static void findAllVars( const string& expr, vector<string>& vars, char start );
        static void findXsYs( const string& expr, vector<string>& xs, vector<string>& ys );

        moose::Parser::value_type Eval( ) const;

        bool IsConstantExpr( const string& expr );

        Parser::varmap_type GetVar() const;

        void SetVarFactory( const char* varName, void* data );

        void DefineConst( const string& constName, moose::Parser::value_type& value );

        void DefineConst( const char* constName, const moose::Parser::value_type& value );

        moose::Parser::value_type Diff( const moose::Parser::value_type a, const moose::Parser::value_type b) const;

        Parser::varmap_type GetConst( ) const;
        Parser::varmap_type GetUsedVar( );
        void ClearVar( );
        const string GetExpr( ) const;
        void SetVarFactory( double* (*fn)(const char*, void*), void *);


    private:
        /* data */
        string expr_;
        moose::Parser::value_type value=0.0;
        Parser::varmap_type var_map_;
        Parser::varmap_type const_map_;
        Parser::varmap_type used_vars_;

        /* Map to variable names and pointer to their values. */
        map<string, double*> map_;

        /* tiny expr */
        vector<te_variable> te_vars_;
        te_expr* te_expr_;
        int* err_ = NULL;
};

} // namespace moose.

#endif /* end of include guard: PARSER_H */
