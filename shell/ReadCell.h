/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

enum ParseStage { COMMENT, DATA, SCRIPT };

class ReadCell
{
	public:
		ReadCell();
		Element* start( const string& cellpath );
		void read( const string& filename, const string& cellpath );
		void readData( const string& line, unsigned int lineNum );
		void readScript( const string& line, unsigned int lineNum );
		void buildCompartment( 
				const string& name, const string& parent,
				double x, double y, double z, double d,
				vector< string >& argv );
	private:
		double RM;
		double CM;
		double RA;
		double EREST_ACT;
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

		Element* lastCompt_;
		bool polarFlag_;
};

