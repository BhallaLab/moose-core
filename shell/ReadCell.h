/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef READCELL_H
#define READCELL_H
enum ParseStage { COMMENT, DATA, SCRIPT };

/**
 * The ReadCell class implements the old GENESIS cellreader
 * functionality. 
 * 
 * ReadCell is partially implemented but works for most common uses and
 * will evolve to some further point.
 *
 * On significant semantic difference from the GENESIS version is that
 * in MOOSE ReadCell can accept values of globals defined in the script,
 * but will NOT alter the script global values.
 */
class ReadCell
{
	public:
		ReadCell( const vector< double >& globalParms );
		Element* start( const string& cellpath );
		void read( const string& filename, const string& cellpath );
		void readData( const string& line, unsigned int lineNum );
		void readScript( const string& line, unsigned int lineNum );
		Element*  buildCompartment( 
				const string& name, const string& parent,
				double x, double y, double z, double d, double& length,
				vector< string >& argv );
		bool buildChannels( 
				Element* compt, vector< string >& argv,
				double diameter, double length);
		Element* findChannel( const string& name );
		bool addChannel( Element* compt, Element* chan, 
				double value, double dia, double length );
		bool addHHChannel( Element* compt, Element* chan, 
				double value, double dia, double length );
		bool addSynChan( Element* compt, Element* chan, 
				double value, double dia, double length );
		bool addSpikeGen( Element* compt, Element* chan, 
				double value, double dia, double length );
		bool addCaConc( Element* compt, Element* chan, 
				double value, double dia, double length );
		bool addNernst( Element* compt, Element* chan, double value );

	private:
		void countProtos( );
		
		double RM_;
		double CM_;
		double RA_;
		double EREST_ACT_;
		double dendrDiam;
		double aveLength;
		double spineSurf;
		double spineDens;
		double spineFreq;
		double membFactor;

		unsigned int numCompartments_;
		unsigned int numChannels_;
		unsigned int numOthers_;

		Element* cell_;

		Element* currCell_;
		Element* lastCompt_;
		Element* protoCompt_;
		unsigned int numProtoCompts_;
		unsigned int numProtoChans_;
		unsigned int numProtoOthers_;

		/** To flag if we are building the main cell, or just a part
		 *  which will be grafted on later.
		 */
		bool graftFlag_;
		bool polarFlag_;
		bool relativeCoordsFlag_;
		vector< Element* > chanProtos_;
};
#endif
