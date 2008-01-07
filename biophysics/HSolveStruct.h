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
	double Gbar_;
	double tau1_;
	double tau2_;
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

/**
 * This struct holds the data structures of the Hines's solver. These are shared
 * by the Hub, Scan and HSolve classes.
 */
struct HSolveStruct
{
	unsigned long            N_;
	vector< unsigned long >  checkpoint_;
	vector< unsigned char >  channelCount_;
	vector< unsigned char >  gateCount_;
	vector< unsigned char >  gateCount1_;
	vector< unsigned char >  gateFamily_;
	vector< double >         M_;
	vector< double >         V_;
	vector< double >         CmByDt_;
	vector< double >         EmByRm_;
	vector< double >         inject_;
	vector< double >         Gbar_;
	vector< double >         GbarEk_;
	vector< double >         state_;
	vector< double >         power_;
	vector< double >         lookup_;
	int                      lookupBlocSize_;
	int                      NDiv_;
	double                   VLo_;
	double                   VHi_;
	double                   dV_;
	
	vector< SpikeGenStruct > spikegen_;
	vector< SynChanStruct > synchan_;
};

#endif // _HSOLVE_STRUCT_H
