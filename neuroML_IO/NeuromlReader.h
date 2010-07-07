/*******************************************************************
 * File:            NeuromlReader.h
 * Description:      
 * Author:          Siji P George
 * E-mail:          siji.suresh@gmail.com
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _NEUROMLREADER_H
#define _NEUROMLREADER_H
#include <neuroML/NCell.h>
#include <neuroML/NBase.h>
class NeuromlReader
{
	public:
		NeuromlReader() {;}
		~NeuromlReader() {;}
		//
         #ifdef USE_NEUROML
		static int targets( Eref object, const string& msg,vector< Eref >& target,const string& type = "" );
		static bool isType( Eref object, const string& type );
		/* Reads in a NeuroML model into MOOSE */
		void  readModel(std::string filename,Id location);
		/* function which insert elements into the vector for setupAlpha */
		void pushtoVector(vector< double >&result,string expr_form,double r,double s,double m);
		/* Returns the surface area of the compartment */
		double calcSurfaceArea(double length,double diameter);
		/* Returns the volume of the compartment */
		double calcVolume(double length,double diameter);
		void setupSynChannels(map< string,vector<string> > &,map< string,vector< string > > &,unsigned int numsynchans);
		void setupChannels(map< string,vector<string> > &,map< string,vector< string > > &,string unit);
		void setupPools(map< string,vector<string> > &,map< string,vector< string > > &,string unit);
	#endif	// USE_NEUROML	
	private:
	#ifdef USE_NEUROML
		Element* compt_;
 		Element* channel_;
		Element* gate_;
		Element* cable_;
		Element* synchannel_;
		Element* leak_;
		Element* ionPool_;
		Element* mgblock_;
		static const double PI;
		map< string,Id > segMap_;
		map< string,string > NMsegMap_;
		 map< string,string >ionchlMap_;
		NCell* ncl_;
	#endif	// USE_NEUROML	
};
extern const Cinfo* initCompartmentCinfo();
extern const Cinfo* initHHChannelCinfo();
extern const Cinfo* initHHGateCinfo();
extern const Cinfo* initInterpolCinfo();
extern const Cinfo* initLeakageCinfo();
extern const Cinfo* initSynChanCinfo();
extern const Cinfo* initCaConcCinfo();
extern const Cinfo* initInterpolCinfo();
extern const Cinfo* initMg_blockCinfo();
#endif // _NEUROMLREADER_H

