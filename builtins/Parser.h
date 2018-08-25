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
#include <map>
#include <exception>

using namespace std;

namespace moose {
    namespace Parser {

        struct ParserException : public std::exception 
        {
            const char* GetMsg()
            {
                return what();
            }
        };

        typedef ParserException exception_type;
    }

template<typename T=double>
class MooseParser
{

    public:
        MooseParser();
        ~MooseParser();

        void DefineVar( const char* varName, T* var)
        {

        }

        void DefineFun( const char* funcName, T (&func)(T) )
        {

        }

        void SetExpr( const string& expr )
        {

        }

        T Eval( map<string, double> map )
        {
            return 0.0;
        }


    private:
        /* data */
        string expr_;
        T value=0.0;
};


}
#endif /* end of include guard: PARSER_H */
