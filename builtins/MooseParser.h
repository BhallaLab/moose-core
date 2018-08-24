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

template<typename T=Parser::value_type>
class MooseParser
{

    public:
        MooseParser() {;}
        ~MooseParser() {;}

        void DefineVar( const char* varName, T* val)
        {
            cout << "NA " << varName << " val: " << val << endl;

        }

        void DefineVar( const string& varName, T& val)
        {
            cout << "NA " << varName << " val: " << val << endl;
        }

        void DefineFun( const char* funcName, T (&func)(T) )
        {

        }

        void SetExpr( const string& expr )
        {
            expr_ = expr;
        }

        T Eval( ) const
        {
            return 0.0;
        }

        Parser::varmap_type GetVar() const
        {
            return var_map_;
        }

        // void SetVarFactory( const string& varName, void* data )
        void SetVarFactory( const char* varName, void* data )
        {
        }

        void DefineConst( const string& constName, T& value )
        {
            const_map_[constName] = value;
        }

        void DefineConst( const char* constName, const T& value )
        {
            const_map_[constName] = value;
        }

        T Diff( const T a, const T b) const
        {
            return a-b;
        }

        Parser::varmap_type GetConst( ) const
        {
            return const_map_;
        }

        Parser::varmap_type GetUsedVar( )
        {
            return used_vars_;
        }

        void ClearVar( )
        {
            const_map_.clear();
            var_map_.clear();
        }

        const string GetExpr( ) const
        {
            return expr_;
        }


    private:
        /* data */
        string expr_;
        T value=0.0;
        Parser::varmap_type var_map_;
        Parser::varmap_type const_map_;
        Parser::varmap_type used_vars_;
};


}
#endif /* end of include guard: PARSER_H */
