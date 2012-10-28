/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MATH_FUNC_TERM_H
#define _MATH_FUNC_TERM_H

/**
 * This is a general FuncTerm for any function of the molecular concs.
 * It uses MathFunc to do the evaluation.
 */
class MathTerm: public FuncTerm
{
	public:
		MathTerm( const vector< unsigned int >& args, MathFunc* func )
			: args_( args ), func_( func )
		{;}

		~MathTerm() {;}

		double operator() ( const double* S, double t ) const;
		unsigned int  getReactants( vector< unsigned int >& molIndex ) const;
		const string& function() const;

	private:
		vector< unsigned int > args_;
		MathFunc* func_;
};

/**
 * This is a general FuncTerm for any function of time and the 
 * molecular concs. It assumes that the first argument of the function is
 * the simulation time. It uses MathFunc to do the evaluation.
 */
class MathTimeTerm: public FuncTerm
{
	public:
		MathTimeTerm( const vector< unsigned int >& args, 
			MathFunc* func )
			: args_( args ), func_( func )
		{;}

		~MathTimeTerm() {;}

		double operator() ( const double* S, double t ) const;
		unsigned int  getReactants( vector< unsigned int >& molIndex ) const;
		const string& function() const;

	private:
		vector< unsigned int > args_;
		MathFunc* func_;
};

#endif // _MATH_FUNC_TERM_H
