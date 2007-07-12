/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**         copyright (C) 2007 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_SCAN_BASE_H
#define _NEURO_SCAN_BASE_H

class NeuroScanBase
{
public:
	NeuroScanBase( HSolveStructure& structure )
	:
		N_( structure.N_ ),
		checkpoint_( structure.checkpoint_ ),
		channelCount_( structure.channelCount_ ),
		gateCount_( structure.gateCount_ ),
		gateCount1_( structure.gateCount1_ ),
		gateFamily_( structure.gateFamily_ ),
		M_( structure.M_ ),
		V_( structure.V_ ),
		CmByDt_( structure.CmByDt_ ),
		EmByRm_( structure.EmByRm_ ),
		inject_( structure.inject_ ),
		Gbar_( structure.Gbar_ ),
		GbarEk_( structure.GbarEk_ ),
		state_( structure.state_ ),
		power_( structure.power_ ),
		lookup_( structure.lookup_ ),
		lookupBlocSize_( structure.lookupBlocSize_ ),
		NDiv_( structure.NDiv_ ),
		VLo_( structure.VLo_ ),
		VHi_( structure.VHi_ ),
		dV_( structure.dV_ )
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
	void initialize( unsigned int seed, double dt );
	
	virtual vector< unsigned int > children(
		unsigned int self, unsigned int parent ) = 0;
	virtual vector< unsigned int > neighbours( unsigned int compartment ) = 0;
	virtual vector< unsigned int > channels( unsigned int compartment ) = 0;
	virtual vector< unsigned int > gates( unsigned int channel ) = 0;
	virtual void field(
		unsigned int object,
		string field,
		double& value ) = 0;
	virtual void rates( unsigned int gate,
		double Vm, double& A, double& B ) = 0;
	
	vector< unsigned int > compartment_;
	vector< unsigned int > channel_;
	vector< unsigned int > gate_;
	
private:
	void constructTree( unsigned int seed );
	void constructMatrix( );
	void constructChannelDatabase( );
	void constructLookupTables( );
	void concludeInit( );
	
	struct CNode
	{
		unsigned int parent_;
		unsigned int self_;
		vector< unsigned int > child_;
		unsigned int state_;
		unsigned int label_;
	};
	
	unsigned long&            N_;
	vector< unsigned long >&  checkpoint_;
	vector< unsigned char >&  channelCount_;
	vector< unsigned char >&  gateCount_;
	vector< unsigned char >&  gateCount1_;
	vector< unsigned char >&  gateFamily_;
	vector< double >&         M_;
	vector< double >&         V_;
	vector< double >&         CmByDt_;
	vector< double >&         EmByRm_;
	vector< double >&         inject_;
	vector< double >&         Gbar_;
	vector< double >&         GbarEk_;
	vector< double >&         state_;
	vector< double >&         power_;
	
	vector< double >&         lookup_;
	int&                      lookupBlocSize_;
	
protected:
	int&                      NDiv_;
	double&                   VLo_;
	double&                   VHi_;
	double&                   dV_;
	double                    dt_;
};

#endif // _NEURO_SCAN_BASE_H
