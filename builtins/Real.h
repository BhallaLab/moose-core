/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _REAL_H
#define _REAL_H

class Real
{
	public: 
		Real();
		Real( double other );
		/*
		void process( const ProcInfo* p, const Eref& e ) {
			;
		}
		*/
		
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		void setThis( double v );
		double getThis() const;

		////////////////////////////////////////////////////////////////
		// Utility stuff
		////////////////////////////////////////////////////////////////
		// const Real& operator=( const Real& other );
		// double operator=( const Real& other );
		// double operator=( const double& other );
	
		////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	private:
		double value_;
};

#endif // _REAL_H
