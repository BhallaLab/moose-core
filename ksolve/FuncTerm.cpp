/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This little class sets up a muParser to execute on entries in the
 * molecule 'n' vector, and possibly on the time t.
 *
 * The user must first set the arg indices (FuncTerm::setArgIndex), before
 * specifying the function string.
 *
 * The arguments are named x0, x1, x2 ..., t )
 *
 */

#include <vector>
#include <sstream>
#include <memory>
using namespace std;

#include "FuncTerm.h"
#include "../utility/numutil.h"
#include "../builtins/MooseParser.h"
#include "../builtins/Variable.h"

FuncTerm::FuncTerm()
    : reactantIndex_( 1, 0 ),
      volScale_( 1.0 ),
      target_( ~0U)
{
    args_ = 0;
}

FuncTerm::~FuncTerm()
{
    if (args_)
    {
        delete[] args_;
    }
}

void FuncTerm::setReactantIndex( const vector< unsigned int >& mol )
{
    reactantIndex_ = mol;
    if ( args_ )
    {
        delete[] args_;
        args_ = 0;
    }

    args_ = new double[ mol.size() + 1 ];
    for ( unsigned int i = 0; i < mol.size(); ++i )
    {
        args_[i] = 0.0;
        addVar( "x"+to_string(i), i );
    }

    // Define a 't' variable even if we don't always use it.
    args_[mol.size()] = 0.0;
    parser_.DefineVar( "t", &args_[mol.size()]);
}

const vector< unsigned int >& FuncTerm::getReactantIndex() const
{
    return reactantIndex_;
}


void showError(moose::Parser::exception_type &e)
{
    cout << "Error occurred in parser.\n"
         << "Message:  " << e.GetMsg() << "\n"
         << endl;
}

void FuncTerm::setExpr( const string& expr )
{
    // Find all variables x\d+ or y\d+ etc, and add them to variable buffer.
    try
    {
        parser_.SetExpr( expr );
        expr_ = expr;
    }
    catch(moose::Parser::exception_type &e)
    {
        showError(e);
        throw(e);
    }
}

const string& FuncTerm::getExpr() const
{
    return expr_;
}

void FuncTerm::setTarget( unsigned int t )
{
    target_ = t;
}

const unsigned int FuncTerm::getTarget() const
{
    return target_;
}

void FuncTerm::setVolScale( double vs )
{
    volScale_ = vs;
}

double FuncTerm::getVolScale() const
{
    return volScale_;
}

const FuncTerm& FuncTerm::operator=( const FuncTerm& other )
{
    args_ = 0; // Don't delete it, the original one is still using it.
    expr_ = other.expr_;
    volScale_ = other.volScale_;
    target_ = other.target_;
    setReactantIndex( other.reactantIndex_ );

    // Parser can't be copied. Copy symbol table and expression.
    parser_.RegisterSymbolTable( other.parser_.GetSymbolTable() );

    return *this;
}

void FuncTerm::addVar( const string& name, size_t i )
{
    parser_.DefineVar( name, &args_[i] );
}

/**
 * This computes the value. The time t is an argument needed by
 * some functions.
 */
double FuncTerm::operator() ( const double* S, double t ) const
{
    if ( !args_ )
        return 0.0;

    for (size_t i = 0; i < reactantIndex_.size(); ++i)
        args_[i] = S[reactantIndex_[i]];

    // Add symbol 't' to the parser.
    args_[reactantIndex_.size()] = t;

    try
    {
        double result = parser_.Eval() * volScale_;
        return result;
    }
    catch (moose::Parser::exception_type &e )
    {
        cerr << "Error: " << e.GetMsg() << endl;
        throw e;
    }

}

void FuncTerm::evalPool( double* S, double t ) const
{
    if ( !args_ || target_ == ~0U )
        return;
    unsigned int i;
    for ( i = 0; i < reactantIndex_.size(); ++i )
        args_[i] = S[reactantIndex_[i]];
    args_[i] = t;

    try
    {
        S[ target_] = parser_.Eval() * volScale_;
    }
    catch ( moose::Parser::exception_type & e )
    {
        showError( e );
        //throw e;
    }
}
