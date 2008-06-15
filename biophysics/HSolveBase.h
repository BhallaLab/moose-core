/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_BASE_H
#define _HSOLVE_BASE_H

/**
 * HSolveBase integrates the equations arising from the compartmental model
 * of a single cell, using the Hines' algorithm.
 */
class HSolveBase
{
public:
	HSolveBase()
	:
		N_( structure_.N_ ),
		checkpoint_( structure_.checkpoint_ ),
		channelCount_( structure_.channelCount_ ),
		M_( structure_.M_ ),
		V_( structure_.V_ ),
		VMid_( structure_.VMid_ ),
		CmByDt_( structure_.CmByDt_ ),
		EmByRm_( structure_.EmByRm_ ),
		inject_( structure_.inject_ ),
		Gk_( structure_.Gk_ ),
		GkEk_( structure_.GkEk_ ),
		state_( structure_.state_ ),
		instant_( structure_.instant_ ),
		lookup_( structure_.lookup_ ),
		channel_( structure_.channel_ ),
		spikegen_( structure_.spikegen_ ),
		synchan_( structure_.synchan_ ),
		caConc_( structure_.caConc_ ),
		ca_( structure_.ca_ ),
		caActivation_( structure_.caActivation_ ),
		caTarget_( structure_.caTarget_ ),
		caDepend_( structure_.caDepend_ )
	{ ; }
	
	HSolveBase& operator=( const HSolveBase& hsb )
	{
		return *this;
	}
	
protected:
	void step( ProcInfo info );
	HSolveStruct structure_;
	
private:
	void calculateChannelCurrents( );
	void updateMatrix( );
	void forwardEliminate( );
	void backwardSubstitute( );
	void advanceCalcium( );
	//~ void advanceChannels( );
	void advanceChannels( double dt );
	void advanceSynChans( ProcInfo info );
	void sendSpikes( ProcInfo info );
	
	unsigned long&            N_;
	vector< unsigned long >&  checkpoint_;
	vector< unsigned char >&  channelCount_;
	vector< double >&         M_;
	vector< double >&         V_;
	vector< double >&         VMid_;
	vector< double >&         CmByDt_;
	vector< double >&         EmByRm_;
	vector< double >&         inject_;
	vector< double >&         Gk_;
	vector< double >&         GkEk_;
	vector< double >&         state_;
	vector< int >&            instant_;
	
	vector< RateLookup >&     lookup_;
	vector< ChannelStruct >&  channel_;
	vector< SpikeGenStruct >& spikegen_;
	vector< SynChanStruct >&  synchan_;
	vector< CaConcStruct >&   caConc_;
	vector< double >&         ca_;
	vector< double >&         caActivation_;
	vector< double* >&        caTarget_;
	vector< double* >&        caDepend_;
	
	static const int INSTANT_X;
	static const int INSTANT_Y;
	static const int INSTANT_Z;
};

#endif // _HSOLVE_BASE_H
