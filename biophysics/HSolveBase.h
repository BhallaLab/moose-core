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
		gateCount_( structure_.gateCount_ ),
		gateCount1_( structure_.gateCount1_ ),
		gateFamily_( structure_.gateFamily_ ),
		M_( structure_.M_ ),
		V_( structure_.V_ ),
		CmByDt_( structure_.CmByDt_ ),
		EmByRm_( structure_.EmByRm_ ),
		inject_( structure_.inject_ ),
		Gbar_( structure_.Gbar_ ),
		GbarEk_( structure_.GbarEk_ ),
		state_( structure_.state_ ),
		power_( structure_.power_ ),
		lookup_( structure_.lookup_ ),
		lookupBlocSize_( structure_.lookupBlocSize_ ),
		VLo_( structure_.VLo_ ),
		dV_( structure_.dV_ )
	{ ; }

	HSolveBase& operator=( const HSolveBase& hsb )
	{
		return *this;
	}
	
protected:
	void step( );
	HSolveStructure          structure_;
	
private:
	void updateMatrix( );
	void forwardEliminate( );
	void backwardSubstitute( );
	void advanceChannels( );
	
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
	double&                   VLo_;
	double&                   dV_;
};

#endif // _HSOLVE_BASE_H
