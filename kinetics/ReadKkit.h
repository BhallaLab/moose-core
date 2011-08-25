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
		double getMaxTime() const;
		double getPlotDt() const;
		double getDefaultVol() const;
		string getBasePath() const;

		//////////////////////////////////////////////////////////////////
		// Undump operations
		//////////////////////////////////////////////////////////////////
		
		void innerRead( ifstream& fin );
		ParseMode readInit( const string& line );
		Id read( const string& filename, const string& cellname, 
			Id parent, const string& solverClass = "Stoich" );
		void readData( const string& line );
		void undump( const vector< string >& args );

		/**
		 * This function sets up the kkit model for a run using the GSL,
		 * which means numerical integration using the GSL, all the plots
		 * specified by the kkit file, and the timestep for plots as 
		 * specified by the kkit file.
		 */
		void setupGslRun();
		void run();
		void dumpPlots( const string& filename );

		//////////////////////////////////////////////////////////////////
		// Building up the model
		//////////////////////////////////////////////////////////////////
		Id buildCompartment( const vector< string >& args );
		Id buildPool( const vector< string >& args );
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
		void buildSumTotal( const string& src, const string& dest );

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
		void separateVols( Id pool, double vol );
		void assignPoolCompartments();
		void assignReacCompartments();
		void assignEnzCompartments();

		//////////////////////////////////////////////////////////////////
		// Utility functions
		//////////////////////////////////////////////////////////////////
		
		/**
		 * Splits up kkit path into head and tail portions, 
		 * tail is returned.
		 * Note that this prepends the basePath to the head.
		 */
		string pathTail( const string& path, string& head ) const;

		/**
		 * make kinetics and graphs elements.
		 */
		void makeStandardElements();

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
		unsigned int numPools_;
		unsigned int numReacs_;
		unsigned int numEnz_;
		unsigned int numMMenz_;
		unsigned int numPlot_;
		unsigned int numOthers_;

		unsigned int lineNum_;

		map< string, int > poolMap_;
		map< string, int > reacMap_;
		map< string, int > enzMap_;
		map< string, int > groupMap_;
		map< string, int > tableMap_;
		map< string, Id > poolIds_;
		map< string, Id > reacIds_;
		map< string, Id > enzIds_;
		map< string, Id > mmEnzIds_;
		map< string, Id > plotIds_;

		/*
		vector< Id > pools_;
		/// This keeps track of all vols, since the pools no longer do.
		vector< double > poolVols_;
		*/

		/// This keeps track of unique volumes
		vector< double > vols_;

		/// List of Ids in each unique volume.
		vector< vector< Id > > volCategories_;
		vector< Id > compartments_;

		Shell* shell_;

		static const double EPSILON;
};

#endif // READ_KKIT_H
