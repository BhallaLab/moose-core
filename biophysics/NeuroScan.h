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
		NeuroScanBase( structure ),
		hub_( structure )
	{
		NDiv_ = 3000;
		VLo_  = -0.100;
		VHi_  = 0.100;
	}
	
	// To keep compiler happy. Should purge it eventually.
	NeuroScan()
	:
		NeuroScanBase( *(new HSolveStruct()) ),
		hub_( *(new HSolveStruct()) )
	{ ; }
  
	// Value Field access function definitions.
	static void setNDiv( const Conn& c, int NDiv );
	static int getNDiv( const Element* e );
	static void setVLo( const Conn& c, double VLo );
	static double getVLo( const Element* e );
	static void setVHi( const Conn& c, double VHi );
	static double getVHi( const Element* e );
	
	// Dest function definitions.
	static void hubCreateFunc( const Conn& c );
	static void readModelFunc( const Conn& c, Element* seed, double dt );
	static void gateFunc( const Conn& c, double A, double B );
	
private:
	void innerHubCreateFunc( Element* e );
	void innerReadModelFunc( Element* e, Element* seed, double dt );
	
	struct GateInfo
	{
		unsigned int chanId;
		unsigned int xIndex;
		unsigned int rIndex;
	};
	
	enum EClass
	{
		COMPARTMENT,
		CHANNEL,
		GATE,
		SPIKEGEN,
		SYNCHAN,
		NONE
	};
	
	/** Portal functions.
	 *  Some of these are not const because they call logElement, which
	 *  maintains local-id <--> global-id mappings.
	 */
	vector< unsigned int > children(
		unsigned int self, unsigned int parent );
	vector< unsigned int > neighbours( unsigned int compartment );
	vector< unsigned int > channels( unsigned int compartment );
	vector< unsigned int > gates( unsigned int channel );
	unsigned int presyn( unsigned int compartment );
	vector< unsigned int > postsyn( unsigned int compartment );
	void field(
		unsigned int object,
		string field,
		double& value );
	void rates( unsigned int gate,
		double Vm, double& A, double& B );
	void synchanFields(
		unsigned int compartment,
		SynChanStruct& scs );
	Element* elm( unsigned int id );
	
	unsigned int logElement(
		Element* el, EClass eclass );
	vector< unsigned int > logElement(
		const vector< Element* >& el, EClass eclass );
	void targets(
		unsigned int id,
		const string& msg,
		vector< Element* >& target ) const;
	EClass type( Element* el );
	
	map< Element*, unsigned int > e2id_;
	vector< Element* > id2e_;
	vector< EClass > eclass_;
	map< unsigned int, GateInfo > gateInfo_;
	Element* scanElm_;
	unsigned int currentChanId_;
	unsigned int currentXIndex_;
	double A_;
	double B_;
	
	NeuroHub hub_;
};

// Used by the solver
extern const Cinfo* initNeuroScanCinfo();

#endif // _NEURO_SCAN_H
