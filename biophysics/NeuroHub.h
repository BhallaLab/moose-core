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
	NeuroHub();
	
	///////////////////////////////////////////////////
	// Field functions
	///////////////////////////////////////////////////
	static unsigned int getNcompt( Eref e );
	
	///////////////////////////////////////////////////
	// Dest functions
	///////////////////////////////////////////////////
	static void compartmentFunc(
		const Conn* c,
		vector< double >* V,
		vector< Element* >* elist );
	static void channelFunc( const Conn* c, vector< Element* >* elist );
	static void spikegenFunc( const Conn* c, vector< Element* >* elist );
	static void synchanFunc( const Conn* c, vector< Element* >* elist );
	static void destroy( const Conn* c );
	static void childFunc( const Conn* c, int stage );
	
	///////////////////////////////////////////////////
	// Field functions (Biophysics)
	///////////////////////////////////////////////////
	static void setCompartmentVm( const Conn* c, double value );
	static double getCompartmentVm( Eref e );
	
	static void setInject( const Conn* c, double value );
	static double getInject( Eref e );
	
	static void setChannelGbar( const Conn* c, double value );
	static double getChannelGbar( Eref e );
	
	static void setSynChanGbar( const Conn* c, double value );
	static double getSynChanGbar( Eref e );
	
	///////////////////////////////////////////////////
	// Dest functions (Biophysics)
	///////////////////////////////////////////////////
	static void comptInjectMsgFunc( const Conn* c, double I );
	
private:
	void innerCompartmentFunc(
		Eref e,
		vector< double >* V,
		vector< Element* >* elist );
	void innerChannelFunc( Eref e, vector< Element* >* elist );
	void innerSpikegenFunc( Eref e, vector< Element* >* elist );
	void innerSynchanFunc( Eref e, vector< Element* >* elist );
	
	static void zombify( 
		Eref hub, Eref e,
		const Finfo* hubFinfo, Finfo* solveFinfo );
	static void unzombify( Element* e );
	static void clearFunc( Eref e );
	static void clearMsgsFromFinfo( Eref e, const Finfo * f );
	static void redirectDestMessages(
		Eref hub, Eref e,
		const Finfo* hubFinfo, const Finfo* eFinfo,
		unsigned int eIndex, vector< unsigned int >& map,
		vector< Element *>* elist, bool retain );
	static void redirectDynamicMessages( Element* e );
	static NeuroHub* getHubFromZombie( Eref e, unsigned int& index );
	
	vector< double >* V_;
	vector< double >* state_;
};

// Used by the scanner
extern const Cinfo* initNeuroHubCinfo();

#endif // _NEURO_HUB_H
