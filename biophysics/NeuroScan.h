/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**         copyright (C) 2007 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_SCAN_H
#define _NEURO_SCAN_H

/**
 * NeuroScan adapts NeuroScanBase into a MOOSE class.
 */
class NeuroScan: public NeuroScanBase
{
public:
	NeuroScan( HSolveStruct& structure )
	:
		NeuroScanBase( structure )
	{
		vDiv_ = 3000;
		vMin_ = -0.100;
		vMax_ = 0.050;
		caDiv_ = 3000;
		caMin_ = 0.0;
		caMax_ = 1000.0;
	}
	
	// To keep compiler happy. Should purge it eventually.
	NeuroScan()
	:
		NeuroScanBase( *(new HSolveStruct()) )
	{ ; }
  
	// Value Field access function definitions.
	static void setVDiv( const Conn* c, int VDiv );
	static int getVDiv( Eref e );
	static void setVMin( const Conn* c, double VMin );
	static double getVMin( Eref e );
	static void setVMax( const Conn* c, double VMax );
	static double getVMax( Eref e );
	static void setCaDiv( const Conn* c, int CaDiv );
	static int getCaDiv( Eref e );
	static void setCaMin( const Conn* c, double CaMin );
	static double getCaMin( Eref e );
	static void setCaMax( const Conn* c, double CaMax );
	static double getCaMax( Eref e );
	
	// Dest function definitions.
	static void hubCreateFunc( const Conn* c );
	static void readModelFunc( const Conn* c, Id seed, double dt );
	static void gateFunc( const Conn* c, double A, double B );
	
private:
	void innerHubCreateFunc( Eref e );
	void innerReadModelFunc( Eref e, Id seed, double dt );
	
	vector< Id > children( Id self, Id parent );
	vector< Id > neighbours( Id compartment );
	vector< Id > channels( Id compartment );
	int gates( Id channel, vector< Id >& );
	Id presyn( Id compartment );
	vector< Id > postsyn( Id compartment );
	int caTarget( Id channel, vector< Id >& );
	int caDepend( Id channel, vector< Id >& );
	void field( Id object, string field, double& value );
	void field( Id object, string field, int& value );
	void rates(
		Id gate,
		const vector< double >& grid,
		vector< double >& A,
		vector< double >& B );
	void synchanFields( Id compartment, SynChanStruct& scs );
	
	int targets(
		Id object,
		const string& msg,
		vector< Id >& target ) const;
	bool isType( Id object, string type );
	
	Eref scanElm_;
	double A_;
	double B_;
	
	NeuroHub hub_;
};

// Used by the solver
extern const Cinfo* initNeuroScanCinfo();

#endif // _NEURO_SCAN_H
