/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_STRUCT_H
#define _HSOLVE_STRUCT_H

typedef double ( *PFDD )( double, double );

struct ChannelStruct
{
public:
	double Gbar_;
	double GbarEk_;
	PFDD takeXpower_;
	PFDD takeYpower_;
	PFDD takeZpower_;
	double Xpower_;
	double Ypower_;
	double Zpower_;
	int instant_;
	
	void setPowers( double Xpower, double Ypower, double Zpower );
	void process( double*& state, double& gk, double& gkek );
	
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
	Element* elm_;
	// SpikeGen fields
	double threshold_;
	double refractT_;
	double amplitude_;
	double state_;
	double lastEvent_;
};

struct SynChanStruct
{
	// Index of parent compartment
	unsigned int compt_;
	Element* elm_;
	// SynChan fields
	double Ek_;
	double Gk_;
double tau1_;
double tau2_;
	double Gbar_;
	double xconst1_;
	double yconst1_;
	double xconst2_;
	double yconst2_;
	double norm_;
	double X_;
	double Y_;
	// The following 3 are still under SynChan's control. Here we simply
	// peek into their values.
	double *activation_;
	double *modulation_;
	priority_queue< SynInfo >* pendingEvents_;
	
	void process( ProcInfo p );
};

struct CaConcStruct
{
	double c_;
	double CaBasal_;
	double factor1_;
	double factor2_;
	
	double process( double activation );
};

/**
 * This struct holds the data structures of the Hines's solver. These are shared
 * by the Hub, Scan and HSolve classes.
 */
struct HSolveStruct
{
	unsigned long             N_;
	vector< unsigned long >   checkpoint_;
	vector< unsigned char >   channelCount_;
	vector< double >          M_;
	vector< double >          V_;
	vector< double >          VMid_;
	vector< double >          CmByDt_;
	vector< double >          EmByRm_;
	vector< double >          inject_;
	vector< double >          Gk_;
	vector< double >          GkEk_;
	vector< double >          state_;
	vector< int >             instant_;
	double                    vMin_;
	double                    vMax_;
	int                       vDiv_;
	double                    caMin_;
	double                    caMax_;
	int                       caDiv_;
	
	vector< RateLookup >      lookup_;
	vector< RateLookupGroup > lookupGroup_;
	vector< ChannelStruct >   channel_;
	vector< SpikeGenStruct >  spikegen_;
	vector< SynChanStruct >   synchan_;
	vector< CaConcStruct >    caConc_;
	vector< double >          ca_;
	vector< double >          caActivation_;
	vector< double* >         caTarget_;
	vector< double* >         caDepend_;
};

#endif // _HSOLVE_STRUCT_H
