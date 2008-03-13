/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Particle_h
#define _Particle_h
class Particle: public Molecule
{
	public:
		Particle();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		static void setPos( const Conn* c, double value, 
				unsigned int i, unsigned int dim );
		static double getPos( const Element* e, unsigned int i, 
				unsigned int dim);
		static void setX( const Conn* c, double val, const unsigned int& i);
		static double getX( const Element* e, const unsigned int& i );
		static void setY( const Conn* c, double val, const unsigned int& i);
		static double getY( const Element* e, const unsigned int& i );
		static void setZ( const Conn* c, double val, const unsigned int& i);
		static double getZ( const Element* e, const unsigned int& i );

		static void setPosVector( const Conn* c, 
			const vector< double >& value, unsigned int dim );
		static vector< double > getPosVector( 
			const Element* e, unsigned int dim);
		static void setXvector( const Conn* c, vector< double > value );
		static vector< double > getXvector( const Element* e );
		static void setYvector( const Conn* c, vector< double > value );
		static vector< double > getYvector( const Element* e );
		static void setZvector( const Conn* c, vector< double > value );
		static vector< double > getZvector( const Element* e );

		//  Override the Molecule operations here
		//  For compatibility I use doubles, but here it is always integral
		static void setN( const Conn* c, double value );
		static double getN( const Element* e );
		static void setConc( const Conn* c, double value );
		static double getConc( const Element* e );
		static void setNinit( const Conn* c, double value );
		static double getNinit( const Element* e );

		static void setD( const Conn* c, double value );
		static double getD( const Element* e );

		// Perhaps have a link to the geom?
		
		///////////////////////////////////////////////////
		// Dest function definitions
		// Most of these are derived right from Molecule, as the
		// particle doesn't do much on its own.
		///////////////////////////////////////////////////
		
		/*
		static void reacFunc( const Conn* c, double A, double B );
		static void sumTotalFunc( const Conn* c, double n );
		void sumProcessFuncLocal( );
		static void sumProcessFunc( const Conn* c, ProcInfo info );
		static void reinitFunc( const Conn* c, ProcInfo info );
		void reinitFuncLocal( Element* e );
		static void processFunc( const Conn* c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );
		*/

	private:
};

// Used by the solver
extern const Cinfo* initParticleCinfo();

#endif // _Particle_h
