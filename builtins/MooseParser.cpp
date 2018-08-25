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
#include "../basecode/global.h"

using namespace std;

#include "MooseParser.h"

#include "../external/tinyexpr/tinyexpr.h"
#include "../utility/print_function.hpp"

void print_tvars( const te_variable* te, size_t n = 1 )
{
    for (size_t i = 0; i < n; i++) 
    {
        auto t = te[i];
        cout << t.name << " at " << t.address << " value " << *((double*)t.address) << endl;
    }
}

namespace moose
{

    MooseParser::MooseParser() 
    {
        te_vars_.clear();

        // Add user defined functions.
        te_variable frnd1 = { "rand", (void*) MooseParser::Rand, TE_FUNCTION0, NULL };
        te_vars_.push_back( frnd1 );

        te_variable frnd2 = { "rand2", (void*) MooseParser::Rand2, TE_FUNCTION2, NULL };
        te_vars_.push_back( frnd2 );

        te_variable ffmod = { "fmod", (void*) MooseParser::Fmod, TE_FUNCTION2, NULL };
        te_vars_.push_back( ffmod );

        // make sure to get this variable right. Every time we set the
        // expression, we clear all value but these.
        num_user_defined_funcs_ = 3;
    }

    MooseParser::~MooseParser() 
    {
        te_free( te_expr_ );
    }

    /*-----------------------------------------------------------------------------
     *  User defined function here.
     *-----------------------------------------------------------------------------*/
    moose::Parser::value_type MooseParser::Rand( )
    {
        return moose::mtrand();
    }

    moose::Parser::value_type MooseParser::Rand2( double a, double b )
    {
        return moose::mtrand( a, b );
    }

    moose::Parser::value_type MooseParser::Fmod( double a, double b )
    {
        return fmod(a, b);
    }


    /*-----------------------------------------------------------------------------
     *  Other function.
     *-----------------------------------------------------------------------------*/
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
        cout << "NA: Defining func " << endl;
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

    void MooseParser::findXsYs( const string& expr, vector<string>& xs, vector<string>& ys )
    {
        findAllVars( expr, xs, 'x' );
        findAllVars( expr, ys, 'y' );
    }

    bool MooseParser::SetExpr( const string& expr )
    {
        if( expr.size() < 1 || expr == "0" || expr == "0.0" )
            return false;

        // NOTE: Before we come here, make sure that map_ is set properly. Map can
        // be set by calling SetVariableMap function.  Following warning is for 
        // developers. It should not be shown to user.
        if( map_.empty() && ! IsConstantExpr(expr) )
        {
            MOOSE_DEBUG( "MOOSE does not yet know where the values of variables x{i}, y{i} etc. " 
                    << " are stored. Did you forget to call SetVariableMap? Doing nothing .." 
                    );
            return false;
        }

        size_t i = 0;

        // Remove all variable after num_user_defined_funcs_ from the vector.
        te_vars_.erase( te_vars_.begin() + num_user_defined_funcs_, te_vars_.end() );
        for (auto itr = map_.begin(); itr != map_.end(); itr++)
        {
            te_variable t;
            t.type = 0;
            t.name = itr->first.c_str();
            t.address = itr->second;
            t.context = NULL;
            te_vars_.push_back( t );
        }


        te_expr_ = te_compile( expr.c_str(), &te_vars_[0], te_vars_.size(), &err_ );
        if( ! te_expr_ )
        {
            printf("Failed to compile:\n\t%s\n", expr.c_str() );
            printf("\t%*s^\nError near here.\n", err_-1, "");
            throw;
        }
        expr_ = expr;
        return true;
    }

    void MooseParser::SetVariableMap( const map<string, double*> m )
    {
        map_.clear();
        for( auto &v : m )
            map_[v.first] = v.second;
    }

    moose::Parser::value_type MooseParser::Eval( ) const
    {
        if( te_expr_ )
            return  te_eval( te_expr_ );
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

    moose::Parser::value_type MooseParser::Diff( 
            const moose::Parser::value_type a, const moose::Parser::value_type b 
            ) const
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
