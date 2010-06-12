/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _M_DOUBLE_H
#define _M_DOUBLE_H

class Mdouble
{
	public: 
		Mdouble();
		Mdouble( double other );
		
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		void setThis( double v );
		double getThis() const;

		////////////////////////////////////////////////////////////////
		// Utility stuff
		////////////////////////////////////////////////////////////////
		// const Mdouble& operator=( const Mdouble& other );
		// double operator=( const Mdouble& other );
		// double operator=( const double& other );
	
		////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	private:
		double value_;
};

#endif // _M_DOUBLE_H
