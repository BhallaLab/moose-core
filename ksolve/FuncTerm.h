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
		virtual ~FuncTerm() {;}
		/**
		 * This computes the value. The time t is an argument because time
		 * might be an argument.
		 */
		virtual double operator() ( double t ) const = 0;

		/**
		 * This function finds the reactant indices in the vector
		 * S. It returns the number of indices found, which are the
		 * entries in molIndex.
		 */
		virtual unsigned int  getReactants( 
			vector< unsigned int >& molIndex,
			const vector< double >& S ) const = 0;
		virtual const string& function() const = 0;
};

class SumTotal: public FuncTerm
{
	public:
		SumTotal( const vector< const double* >& mol )
			: mol_( mol )
		{;}

		double operator() ( double t ) const;
		unsigned int  getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const;
		const string& function() const;

	private:
		vector< const double* > mol_;
};
