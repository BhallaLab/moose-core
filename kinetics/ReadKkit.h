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
		Id buildGraph( const vector< string >& args );
		Id buildGeometry( const vector< string >& args );
		Id buildInfo( Id parent, map< string, int >& m, 
			const vector< string >& args );

		//////////////////////////////////////////////////////////////////
		// Special ops in the model definition
		//////////////////////////////////////////////////////////////////
		void addmsg( const vector< string >& args );
		void innerAddMsg( 
			const string& src, const map< string, Id >& m1, 
				const string& srcMsg,
			const string& dest, const map< string, Id >& m2, 
				const string& destMsg );
		void call( const vector< string >& args );
		void objdump( const vector< string >& args );
		void textload( const vector< string >& args );
		void loadtab( const vector< string >& args );
		void separateVols( Id mol, double vol );
		void assignCompartments();

		//////////////////////////////////////////////////////////////////
		// Utility functions
		//////////////////////////////////////////////////////////////////
		
		/**
		 * Splits up kkit path into head and tail portions, 
		 * tail is returned.
		 * Note that this prepends the basePath to the head.
		 */
		string pathTail( const string& path, string& head ) const;

	private:
		string basePath_; /// Base path into which entire kkit model will go
		Id baseId_; /// Base Id onto which entire kkit model will go.

		double fastdt_; /// fast numerical timestep from kkit.
		double simdt_;	/// regular numerical timestep from kkit.
		double controldt_;	/// Timestep for updating control graphics
		double plotdt_;		/// Timestep for updating plots
		double maxtime_;	/// Simulation run time.
		double transientTime_;	/// Time to run model at fastdt
		bool useVariableDt_;	/// Use both fast and sim dts.
		double defaultVol_;		/// Default volume for new compartments.
		unsigned int version_;	/// KKit version.
		unsigned int initdumpVersion_;	/// Initdump too has a version.

		unsigned int numCompartments_;
		unsigned int numMols_;
		unsigned int numReacs_;
		unsigned int numEnz_;
		unsigned int numMMenz_;
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
		map< string, Id > mmEnzIds_;
		map< string, Id > plotIds_;

		vector< double > vols_;
		vector< vector< Id > > mols_;
		vector< Id > compartments_;

		Shell* shell_;

		static const double EPSILON;
};

#endif // READ_KKIT_H
