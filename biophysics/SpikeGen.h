/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SpikeGen_h
#define _SpikeGen_h

class SpikeGen
{
  public:
    SpikeGen():threshold_(0.0),
               refractT_(0.0),
               amplitude_(1.0),
               state_(0.0),
               lastEvent_(0.0),
               V_(0.0),
               edgeTriggered_(1)
    {
    }

	//////////////////////////////////////////////////////////////////
	// Field functions.
	//////////////////////////////////////////////////////////////////
		static void setThreshold( const Conn* c, double threshold );
		static double getThreshold( Eref e );

		static void setRefractT( const Conn* c, double val );
		static double getRefractT( Eref e );

		static void setAmplitude( const Conn* c, double val );
		static double getAmplitude( Eref e );

		static void setState( const Conn* c, double val );
		static double getState( Eref e );

                static void setEdgeTriggered( const Conn* c, int yes);
                static int getEdgeTriggered( Eref e);

	//////////////////////////////////////////////////////////////////
	// Message dest functions.
	//////////////////////////////////////////////////////////////////

    virtual void innerProcessFunc( const Conn* c, ProcInfo p );
	static void processFunc( const Conn* c, ProcInfo p );
	static void reinitFunc( const Conn* c, ProcInfo p );
	static void VmFunc( const Conn* c, double val );

	protected:
		double threshold_;
		double refractT_;
		double amplitude_;
		double state_;
		double lastEvent_;
		double V_;
                bool fired_;
                int edgeTriggered_;
};

// Used by solver, readcell, etc.
extern const Cinfo* initSpikeGenCinfo();

#endif // _SpikeGen_h
