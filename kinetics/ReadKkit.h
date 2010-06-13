/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _READ_KKIT_H
#define _READ_KKIT_H

class ReadKkit
{
	public: 
		enum ParseMode {
			DATA,
			INIT,
			COMMENT,
			LINE_CONTINUE
		};

		ReadKkit();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////


		//////////////////////////////////////////////////////////////////
		// Undump operations
		//////////////////////////////////////////////////////////////////
		
		Id findParent( const string& path ) const;
		void innerRead( ifstream& fin );
		ParseMode readInit( const string& line );
		void read( const string& filename, const string& cellname, 
			Id parent );
		void readData( const string& line );
		void undump( const vector< string >& args );


		//////////////////////////////////////////////////////////////////
		// Building up the model
		//////////////////////////////////////////////////////////////////
		Id buildCompartment( const vector< string >& args );
		Id buildMol( const vector< string >& args );
		Id buildReac( const vector< string >& args );
		Id buildEnz( const vector< string >& args );
		Id buildPlot( const vector< string >& args );
		Id buildTab( const vector< string >& args );
		unsigned int loadTab(  const vector< string >& args );
		Id buildGroup( const vector< string >& args );
		Id buildText( const vector< string >& args );

		//////////////////////////////////////////////////////////////////
		// Special ops in the model definition
		//////////////////////////////////////////////////////////////////
		void addmsg( const vector< string >& args );
		void call( const vector< string >& args );
		void objdump( const vector< string >& args );
		void textload( const vector< string >& args );
		void loadtab( const vector< string >& args );

		// static const Cinfo* initCinfo();
	private:
		double fastdt_;
		double simdt_;
		double controldt_;
		double plotdt_;
		double maxtime_;
		double transientTime_;
		bool useVariableDt_;
		double defaultVol_;
		unsigned int version_;
		unsigned int initdumpVersion_;

		unsigned int numCompartments_;
		unsigned int numMols_;
		unsigned int numReacs_;
		unsigned int numEnz_;
		unsigned int numPlot_;
		unsigned int numOthers_;

		unsigned int lineNum_;

		map< string, int > molMap_;
		map< string, int > reacMap_;
		map< string, int > enzMap_;
		map< string, int > groupMap_;
		map< string, int > tableMap_;
		map< string, Id > molIds_;
		map< string, Id > reacIds_;
		map< string, Id > enzIds_;
};

#endif // READ_KKIT_H
