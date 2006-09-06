/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Stoich_h
#define _Stoich_h
class Stoich
{
	friend class StoichWrapper;
	public:
		Stoich()
		{
			nMols_ = 0;
			nVarMols_ = 0;
			nSumTot_ = 0;
			nBuffered_ = 0;
			nReacs_ = 0;
			nEnz_ = 0;
			nMmEnz_ = 0;
			nExternalRates_ = 0;
			useOneWayReacs_ = 0;
		}

	private:
		int nMols_;
		int nVarMols_;
		int nSumTot_;
		int nBuffered_;
		int nReacs_;
		int nEnz_;
		int nMmEnz_;
		int nExternalRates_;
		int useOneWayReacs_;
		string path_;
		vector< double > S_; 	
		vector< double > Sinit_; 	
		vector< double > v_;	
		vector< RateTerm* > rates_; 
		vector< int > sumTotals_;
		SparseMatrix N_; 
		vector< int > path2mol_;
		vector< int > mol2path_;
		void updateRates( vector< double>* yprime, double dt  );
		void updateV( );
};
#endif // _Stoich_h
