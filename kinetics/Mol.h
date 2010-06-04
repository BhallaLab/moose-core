/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Mol: public Data
{
	friend void testSyncArray( unsigned int size, unsigned int numThreads,
		unsigned int method );
	friend void checkVal( double time, const Mol* m, unsigned int size );
	friend void forceCheckVal( double time, Element* e, unsigned int size );

	public: 
		Mol()
			: n_( 0.0 ), nInit_( 0.0 ), A_( 0.0 ), B_( 0.0 )
			{;}

		Mol( double nInit )
			: n_( 0.0 ), nInit_( nInit ), A_( 0.0 ), B_( 0.0 )
			{;}
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setN( double v );
		double getN() const;
		void setNinit( double v );
		double getNinit() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void eprocess( Eref e, const Qinfo* q, ProcInfo* p );
		void reinit( const Eref& e, const Qinfo*q, ProcInfo* p );
		void reac( double A, double B );
		void sumTotal( double v );

		static const Cinfo* initCinfo();
	private:
		double n_;
		double nInit_;
		double A_;
		double B_;
};
