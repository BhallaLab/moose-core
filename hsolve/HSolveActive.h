/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_ACTIVE_H
#define _HSOLVE_ACTIVE_H

class HSolveActive: public HSolvePassive
{
	typedef vector< CurrentStruct >::iterator currentVecIter;

public:
	HSolveActive();
	
	void setup( Id seed, double dt );
	void step( ProcPtr info );
	void reinit();
	
protected:
	/**
	 * Solver parameters: exposed as fields in MOOSE
	 */
	
	/**
	 * caAdvance_: This flag determines how current flowing into a calcium pool
	 * is computed. A value of 0 means that the membrane potential at the
	 * beginning of the time-step is used for the calculation. This is how
	 * GENESIS does its computations. A value of 1 means the membrane potential
	 * at the middle of the time-step is used. This is the correct way of
	 * integration, and is the default way.
	 */
	int                       caAdvance_;
	
	/**
	 * vMin_, vMax_, vDiv_,
	 * caMin_, caMax_, caDiv_:
	 * 
	 * These are the parameters for the lookup tables for rate constants.
	 * 'min' and 'max' are the boundaries within which the function is defined.
	 * 'div' is the number of divisions between min and max.
	 */
	double                    vMin_;
	double                    vMax_;
	int                       vDiv_;
	double                    caMin_;
	double                    caMax_;
	int                       caDiv_;
	
	/**
	 * Internal data structures. Will also be accessed in derived class HSolve.
	 */
	vector< CurrentStruct >   current_;
	vector< double >          state_;
	//~ vector< int >             instant_;
	vector< ChannelStruct >   channel_;
	vector< SpikeGenStruct >  spikegen_;
	vector< SynChanStruct >   synchan_;
	vector< CaConcStruct >    caConc_;
	vector< double >          ca_;
	vector< double >          caActivation_;
	vector< double* >         caTarget_;
	LookupTable               vTable_;
	LookupTable               caTable_;
	vector< bool >            gCaDepend_;
	vector< unsigned int >    caCount_;
	vector< int >             caDependIndex_;
	vector< LookupColumn >    column_;
	vector< LookupRow >       caRowCompt_;
	vector< LookupRow* >      caRow_;
	vector< int >             channelCount_;
	vector< currentVecIter >  currentBoundary_;
	vector< unsigned int >    chan2compt_;
	vector< unsigned int >    chan2state_;
	vector< double >          externalCurrent_;
	vector< Id >              caConcId_;
	vector< Id >              channelId_;
	vector< Id >              gateId_;
	//~ vector< vector< Id > >    externalChannelId_;
	
private:
	/**
	 * Setting up of data structures: Defined in HSolveActiveSetup.cpp
	 */
	void readHHChannels();
	void readGates();
	void readCalcium();
	void readSynapses();
	void readExternalChannels();
	
	void createLookupTables();
	void cleanup();
	
	/**
	 * Reinit code: Defined in HSolveActiveSetup.cpp
	 */
	void reinitCompartments();
	void reinitCalcium();
	void reinitChannels();
	
	/**
	 * Integration: Defined in HSolveActive.cpp
	 */
	void calculateChannelCurrents();
	void updateMatrix();
	void forwardEliminate();
	void backwardSubstitute();
	void advanceCalcium();
	void advanceChannels( double dt );
	void advanceSynChans( ProcPtr info );
	void sendSpikes( ProcPtr info );
	void sendValues( ProcPtr info );
	
	static const int INSTANT_X;
	static const int INSTANT_Y;
	static const int INSTANT_Z;
};

#endif // _HSOLVE_ACTIVE_H
