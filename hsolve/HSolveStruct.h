/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_STRUCT_H
#define _HSOLVE_STRUCT_H

typedef double ( *PFDD )( double, double );

struct CompartmentStruct
{
	double CmByDt;
	double EmByRm;
};

struct InjectStruct
{
	InjectStruct()
		:
		injectVarying( 0.0 ),
		injectBasal( 0.0 )
	{ ; }
	
	double injectVarying;
	double injectBasal;
};

struct CurrentStruct
{
	double Gk;
	double Ek;
};

struct ChannelStruct
{
public:
	double Gbar_;
	PFDD takeXpower_;
	PFDD takeYpower_;
	PFDD takeZpower_;
	double Xpower_;
	double Ypower_;
	double Zpower_;
	int instant_;
	
	void setPowers( double Xpower, double Ypower, double Zpower );
	void process( double*& state, CurrentStruct& current );
	
private:
	static PFDD selectPower( double power );
	
	static double power1( double x, double p ) {
		return x;
	}
	static double power2( double x, double p ) {
		return x * x;
	}
	static double power3( double x, double p ) {
		return x * x * x;
	}
	static double power4( double x, double p ) {
		return power2( x * x, p );
	}
	static double powerN( double x, double p );
};

struct SpikeGenStruct
{
	// Index of parent compartment
	unsigned int compt_;
	Id elm_;
};

struct SynChanStruct
{
	// Index of parent compartment
	unsigned int compt_;
	Id elm_;
};

struct CaConcStruct
{
	double c_;
	double CaBasal_;
	double factor1_;
	double factor2_;
	double process( double activation );
};

#endif // _HSOLVE_STRUCT_H
