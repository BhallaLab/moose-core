/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_HUB_H
#define _NEURO_HUB_H

/**
 * Biophysical elements in a neuronal model hand over control (computation,
 * fields, messages) to the solver. NeuroHub handles the fields and messages--
 * it can do so because the solver's data structures are shared between the
 * integrator, hub and the scanner.
 */
class NeuroHub
{
public:
	NeuroHub( HSolveStruct& structure )
	:
		M_( structure.M_ ),
		V_( structure.V_ ),
		CmByDt_( structure.CmByDt_ ),
		EmByRm_( structure.EmByRm_ ),
		inject_( structure.inject_ ),
		Gbar_( structure.Gbar_ ),
		GbarEk_( structure.GbarEk_ ),
		state_( structure.state_ ),
		power_( structure.power_ )
	{ ; }
	
	// To keep compiler happy. Should purge it eventually.
	NeuroHub()
	:
		M_( *(new std::vector<double>()) ),
		V_( *(new std::vector<double>()) ),
		CmByDt_( *(new std::vector<double>()) ),
		EmByRm_( *(new std::vector<double>()) ),
		inject_( *(new std::vector<double>()) ),
		Gbar_( *(new std::vector<double>()) ),
		GbarEk_( *(new std::vector<double>()) ),
		state_( *(new std::vector<double>()) ),
		power_( *(new std::vector<double>()) )
	{ ; }
	
	///////////////////////////////////////////////////
	// Default assignment operator needed since we have
	// reference members. Currently just returns self.
	///////////////////////////////////////////////////
	NeuroHub& operator=( const NeuroHub& nh )
	{
		return *this;
	}
	
	///////////////////////////////////////////////////
	// Field functions
	///////////////////////////////////////////////////
	static unsigned int getNcompt( const Element* e );
	
	///////////////////////////////////////////////////
	// Dest functions
	///////////////////////////////////////////////////
	static void compartmentFunc( const Conn* c, vector< Element* >* elist );
	static void channelFunc( const Conn* c, vector< Element* >* elist );
	static void spikegenFunc( const Conn* c, vector< Element* >* elist );
	static void synchanFunc( const Conn* c, vector< Element* >* elist );
	static void destroy( const Conn* c );
	static void childFunc( const Conn* c, int stage );
	
	///////////////////////////////////////////////////
	// Field functions (Biophysics)
	///////////////////////////////////////////////////
	static void setComptVm( const Conn* c, double value );
	static double getComptVm( const Element* e );
	
	static void setInject( const Conn* c, double value );
	static double getInject( const Element* e );
	
	static void setChanGbar( const Conn* c, double value );
	static double getChanGbar( const Element* e );
	
	static void setSynChanGbar( const Conn* c, double value );
	static double getSynChanGbar( const Element* e );
	///////////////////////////////////////////////////
	// Dest functions (Biophysics)
	///////////////////////////////////////////////////
	static void comptInjectMsgFunc( const Conn* c, double I );
	
private:
	void innerCompartmentFunc( Element* e,
		vector< Element* >* elist );
	void innerChannelFunc( Element* e,
		vector< Element* >* elist );
	void innerSpikegenFunc( Element* e,
		vector< Element* >* elist );
	void innerSynchanFunc( Element* e,
		vector< Element* >* elist );
	
	static void zombify( 
		Element* hub, Element* e,
		const Finfo* hubFinfo, Finfo* solveFinfo );
	static void unzombify( const Conn* c );
	static void clearFunc( const Conn* c );
	static void redirectDestMessages(
		Element* hub, Element* e,
		const Finfo* hubFinfo, const Finfo* eFinfo,
		unsigned int eIndex, vector< unsigned int >* map );
	static void redirectDynamicMessages( Element* e );
	static NeuroHub* getHubFromZombie(
		const Element* e, const Finfo* srcFinfo,
		unsigned int& index );
	
	vector< double >&        M_;
	vector< double >&        V_;
	vector< double >&        CmByDt_;
	vector< double >&        EmByRm_;
	vector< double >&        inject_;
	vector< double >&        Gbar_;
	vector< double >&        GbarEk_;
	vector< double >&        state_;
	vector< double >&        power_;
};

// Used by the scanner
extern const Cinfo* initNeuroHubCinfo();

#endif // _NEURO_HUB_H
