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
	void solve( ProcInfo info );
	
	/**
	 * Interface functions to access the solver's data: Defined in HSolveInterface.cpp
	 */
	
	/// Interface to compartments
	const vector< Id >& getCompartments( ) const;
	double getVm( unsigned int index ) const;
	void setVm( unsigned int index, double value );
	double getInject( unsigned int index ) const;
	void setInject( unsigned int index, double value );
	double getIm( unsigned int index ) const;
	void addInject( unsigned int index, double value );
	void addGkEk( unsigned int index, double v1, double v2 );
	
	/// Interface to channels
	const vector< Id >& getHHChannels( ) const;
	double getHHChannelGbar( unsigned int index ) const;
	void setHHChannelGbar( unsigned int index, double value );
	double getEk( unsigned int index ) const;
	void setEk( unsigned int index, double value );
	double getGk( unsigned int index ) const;
	void setGk( unsigned int index, double value );
	// Ik is read-only
	double getIk( unsigned int index ) const;
	double getX( unsigned int index ) const;
	void setX( unsigned int index, double value );
	double getY( unsigned int index ) const;
	void setY( unsigned int index, double value );
	double getZ( unsigned int index ) const;
	void setZ( unsigned int index, double value );
	
	/// Interface to CaConc
	const vector< Id >& getCaConcs( ) const;
	double getCaBasal( unsigned int index ) const;
	void setCaBasal( unsigned int index, double value );
	double getCa( unsigned int index ) const;
	void setCa( unsigned int index, double value );
	
	/// Interface to external channels
	const vector< vector< Id > >& getExternalChannels( ) const;
	
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

private:
	/**
	 * Internal data structures
	 */
	vector< CurrentStruct >   current_;
	vector< double >          state_;
	vector< int >             instant_;
	vector< ChannelStruct >   channel_;
	vector< SpikeGenStruct >  spikegen_;
	vector< SynChanStruct >   synchan_;
	vector< CaConcStruct >    caConc_;
	vector< double >          caActivation_;
	vector< CaTractStruct >   caTract_;
	vector< CurrentStruct* >  caSource_;
	vector< double* >         caTarget_;
	LookupTable               vTable_;
	LookupTable               caTable_;
	vector< Id >              caConcId_;
	vector< bool >            gCaDepend_;
	vector< int >             caDependIndex_;
	vector< LookupColumn >    column_;
	vector< LookupRow >       caRow_;
	vector< LookupRow* >      caRowChan_;
	vector< Id >              channelId_;
	vector< Id >              gateId_;
	vector< int >             channelCount_;
	vector< currentVecIter >  currentBoundary_;
	vector< unsigned int >    chan2compt_;
	vector< unsigned int >    chan2state_;
	vector< vector< Id > >    externalChannelId_;
	vector< double >          externalCurrent_;
	
	/**
	 * Setting up of data structures: Defined in HSolveActiveSetup.cpp
	 */
	void readHHChannels( );
	void readGates( );
	void readCalcium( );
	void readSynapses( );
	void readExternalChannels( );
	void createLookupTables( );
	void cleanup( );

	/**
	 * Integration: Defined in HSolveActive.cpp
	 */
	void calculateChannelCurrents( );
	void updateMatrix( );
	void forwardEliminate( );
	void backwardSubstitute( );
	void advanceCalcium( );
	void advanceChannels( double dt );
	void advanceSynChans( ProcInfo info );
	void sendSpikes( ProcInfo info );
	void sendValues( );
	
	static const int INSTANT_X;
	static const int INSTANT_Y;
	static const int INSTANT_Z;
};

#endif // _HSOLVE_ACTIVE_H
