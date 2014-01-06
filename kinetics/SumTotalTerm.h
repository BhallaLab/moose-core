/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This is a special FuncTerm that returns the sum of the molecular concs.
 * Does the calculation directly, so it is fast.
 */
#ifndef _SUM_TOTAL_TERM_H
#define _SUM_TOTAL_TERM_H
class SumTotalTerm: public FuncTerm
{
	public:
		SumTotalTerm() {;}
		~SumTotalTerm() {;}

		double operator() ( const double* S, double t ) const;
		unsigned int  getReactants( vector< unsigned int >& molIndex) const;
		void setReactants( const vector< unsigned int >& mol );
		const string& function() const;

	private:
		vector< unsigned int > mol_;
};


#endif // _SUM_TOTAL_TERM_H
