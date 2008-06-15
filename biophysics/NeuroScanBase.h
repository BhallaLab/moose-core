/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_SCAN_BASE_H
#define _NEURO_SCAN_BASE_H

/**
 * Component of the solver which reads in a neuronal model into the solver's
 * data structures.
 */
class NeuroScanBase
{
public:
	NeuroScanBase( HSolveStruct& structure )
	:
		N_( structure.N_ ),
		checkpoint_( structure.checkpoint_ ),
		channelCount_( structure.channelCount_ ),
		M_( structure.M_ ),
		V_( structure.V_ ),
		VMid_( structure.VMid_ ),
		CmByDt_( structure.CmByDt_ ),
		EmByRm_( structure.EmByRm_ ),
		inject_( structure.inject_ ),
		state_( structure.state_ ),
		instant_( structure.instant_ ),
		lookupGroup_( structure.lookupGroup_ ),
		lookup_( structure.lookup_ ),
		vDiv_( structure.vDiv_ ),
		vMin_( structure.vMin_ ),
		vMax_( structure.vMax_ ),
		caDiv_( structure.caDiv_ ),
		caMin_( structure.caMin_ ),
		caMax_( structure.caMax_ ),
		channel_( structure.channel_ ),
		spikegen_( structure.spikegen_ ),
		synchan_( structure.synchan_ ),
		caConc_( structure.caConc_ ),
		ca_( structure.ca_ ),
		caActivation_( structure.caActivation_ ),
		caTarget_( structure.caTarget_ ),
		caDepend_( structure.caDepend_ )
	{ ; }
	
	virtual ~NeuroScanBase()
	{
		;
	}
	
	///////////////////////////////////////////////////
	// Default assignment operator needed since we have
	// reference members. Currently just returns self.
	///////////////////////////////////////////////////
	NeuroScanBase& operator=( const NeuroScanBase& nsb )
	{
		return *this;
	}
	
protected:
	void initialize( Id seed, double dt );
	
	virtual vector< Id > children( Id self, Id parent ) = 0;
	virtual vector< Id > neighbours( Id compartment ) = 0;
	virtual vector< Id > channels( Id compartment ) = 0;
	virtual int gates( Id channel, vector< Id >& ) = 0;
	virtual Id presyn( Id compartment ) = 0;
	virtual vector< Id > postsyn( Id compartment ) = 0;
	virtual int caTarget( Id channel, vector< Id >& ) = 0;
	virtual int caDepend( Id channel, vector< Id >& ) = 0;
	virtual void field( Id object, string field, double& value ) = 0;
	virtual void field( Id object, string field, int& value ) = 0;
	virtual void rates(
		Id gate,
		const vector< double >& grid,
		vector< double >& A,
		vector< double >& B ) = 0;
	virtual void synchanFields( Id synchan, SynChanStruct& scs ) = 0;
	
	vector< Id > compartmentId_;
	vector< Id > channelId_;
	vector< Id > gateId_;
	vector< bool > gCaDepend_;
	
private:
	void readCompartments( Id seed );
	void readChannels( );
	void readGates( );
	void readCalcium( );
	void readSynapses( );
	void createMatrix( );
	void createLookupTables( );
	void concludeInit( );
	
	struct CNode
	{
		Id parent_;
		Id self_;
		vector< Id > child_;
		unsigned int state_;
		unsigned int label_;
	};
	
protected:
	unsigned long&             N_;
	vector< unsigned long >&   checkpoint_;
	vector< unsigned char >&   channelCount_;
	vector< double >&          M_;
	vector< double >&          V_;
	vector< double >&          VMid_;
	vector< double >&          CmByDt_;
	vector< double >&          EmByRm_;
	vector< double >&          inject_;
	vector< double >&          state_;
	vector< int >&             instant_;
	
	vector< RateLookupGroup >& lookupGroup_;
	vector< RateLookup >&      lookup_;
	
	int&                       vDiv_;
	double&                    vMin_;
	double&                    vMax_;
	int&                       caDiv_;
	double&                    caMin_;
	double&                    caMax_;
	double                     dt_;
	vector< ChannelStruct >&   channel_;
	vector< SpikeGenStruct >&  spikegen_;
	vector< SynChanStruct >&   synchan_;
	vector< CaConcStruct >&    caConc_;
	vector< double >&          ca_;
	vector< double >&          caActivation_;
	vector< double* >&         caTarget_;
	vector< double* >&         caDepend_;
};

#endif // _NEURO_SCAN_BASE_H
