/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FUNC_TERM_H
#define _FUNC_TERM_H

class FuncTerm
{
	public:
		FuncTerm() {;}
		virtual ~FuncTerm()
		{;}
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
		virtual void setReactants( const vector< unsigned int >& mol ) = 0;
		virtual unsigned int  getReactants( 
			vector< unsigned int >& molIndex ) const = 0;
		virtual const string& function() const = 0;
};

#endif // _FUNC_TERM_H
