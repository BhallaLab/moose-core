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
#include "../utility/testing_macros.hpp"
#include "../utility/utility.h"


FuncTerm::FuncTerm()
    : reactantIndex_( 1, 0 ),
      volScale_( 1.0 ),
      target_( ~0U)
{
    args_ = unique_ptr<double[]>(nullptr);
}

FuncTerm::~FuncTerm()
{
}

void FuncTerm::setReactantIndex(const vector<unsigned int>& mol)
{
    reactantIndex_ = mol;

    // The address of args_ has changed now. Any previous mapping with ExprTK
    // symbol table is now invalidated and thus can't be used anymore. We need
    // to re-assign parser as well
    // The address of args_ has changed now. Any previous mapping with ExprTK
    // symbol table is now invalidated and thus can't be used anymore. We need
    // to re-assign parser.
    parser_.Reinit();
    args_ = unique_ptr<double[]>(new double[mol.size()+1]());
    for ( unsigned int i = 0; i < mol.size(); ++i )
        addVar( "x"+to_string(i), i );

    // Define a 't' variable even if we don't always use it.
    addVar( "t", mol.size());

    // Need to compile else we get garbage value. Not sure why ExprTK is so
    // finicy about it.
    parser_.CompileExpr();
}

const vector<unsigned int>& FuncTerm::getReactantIndex() const
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
    args_.reset( other.args_.get() ); // unique_ptr
    expr_ = other.expr_;
    volScale_ = other.volScale_;
    target_ = other.target_;
    setReactantIndex( other.reactantIndex_ );
    return *this;
}

void FuncTerm::addVar( const string& name, size_t i )
{
    parser_.DefineVar(name, args_[i]);
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

    // update value of t.
    args_[reactantIndex_.size()] = t;

#ifdef exprtk_enable_debugging
    cout << "FuncTerm::operator() :: ";
    for (size_t i = 0; i < reactantIndex_.size(); i++)
      cout << args_[i] << "(" << reactantIndex_[i] << "), ";
    cout << args_[reactantIndex_.size()] << endl;
#endif

    try
    {
        double result = parser_.Eval() * volScale_;
#ifdef exprtk_enable_debugging
        cout << " Result= " << result << endl;
#endif
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

    size_t i;
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
