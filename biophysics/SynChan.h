#ifndef _SynChan_h
#define _SynChan_h
/************************************************************************ This program is part of 'MOOSE', the** Messaging Object Oriented Simulation Environment,** also known as GENESIS 3 base code.**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS** It is made available under the terms of the** GNU Lesser General Public License version 2.1** See the file COPYING.LIB for the full notice.**********************************************************************/class SynChan
{
	friend class SynChanWrapper;
	public:
		SynChan()
			: Ek_( 0.0 ), Gbar_( 0.0 ), 
			tau1_( 1.0e-3 ), tau2_( 1.0e-3 ),
			normalizeWeights_( 0 )
		{
			;
		}

	private:
		double Ek_;
		double Gk_;
		double Ik_;
		double Gbar_;
		double tau1_;
		double tau2_;
		int normalizeWeights_;
		double xconst1_;
		double yconst1_;
		double xconst2_;
		double yconst2_;
		double norm_;
		double activation_;
		double modulation_;
		double X_;	
		double Y_;	
		priority_queue< SynInfo > pendingEvents_;
};
#endif // _SynChan_h
