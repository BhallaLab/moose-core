/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class FuncTerm
{
	public:
		FuncTerm() {;}
		virtual ~FuncTerm();
		/**
		 * This computes the value. The time t is an argument needed by
		 * some peculiar functions.
		 */
		virtual double operator() ( const double* S, double t ) const = 0;

		/**
		 * This function finds the reactant indices in the vector
		 * S. It returns the number of indices found, which are the
		 * entries in molIndex.
		 */
		virtual unsigned int  getReactants( 
			vector< unsigned int >& molIndex ) const = 0;
		virtual const string& function() const = 0;
};

/**
 * This is a special FuncTerm that returns the sum of the molecular concs.
 * Does the calculation directly, so it is fast.
 */
class SumTotal: public FuncTerm
{
	public:
		SumTotal( const vector< unsigned int >& mol )
			: mol_( mol )
		{;}

		~SumTotal() {;}

		double operator() ( const double* S, double t ) const;
		unsigned int  getReactants( vector< unsigned int >& molIndex) const;
		const string& function() const;

	private:
		vector< unsigned int > mol_;
};

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

/**
 * This is a general FuncTerm for any function of time and a delay
 * term for each of the molecular args. If the delay is zero, it uses
 * the mol conc directly, otherwise utilizes a ring buffer with the 
 * specified time resolution. Nasty stuff.
 */
