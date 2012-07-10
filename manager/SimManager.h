/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class SimManager
{
	public:
		enum TreeType { CHEM_ONLY, KKIT, CHEM_SPACE, CHEM_SPACE_MULTISOLVER, SIGNEUR };

		SimManager();
		~SimManager();

		// Field Definitions
		void setAutoPlot( bool v );
		bool getAutoPlot() const;

		void setSyncTime( double v );
		double getSyncTime() const;

		void setPlotDt( double v );
		double getPlotDt() const;
		void setSimDt( double v );
		double getSimDt() const;
		void setRunTime( double v );
		double getRunTime() const;
		void setVersion( unsigned int v );
		unsigned int getVersion() const;
		void setMethod( string v );
		string getMethod() const;

		/// Destination function
		void build( const Eref& e, const Qinfo* q, string method );


		/**
		 * Builds standard kinetic model tree. It looks like:
		 *
		 * /base   :                   		Manager
    	 * 		/<solver_name>              Solver
    	 * 		/<compartment_name>         Mesh
         * 			/mesh               		MeshEntry
         * 			/boundary           		Boundary
         * 			/<poolname>         		Pool
         * 				/<enzname>      		MMEnz
         * 				/<enzname>      		Enz
         * 					/<enzname_cplx> 	Pool
         * 			/<reacname>         		Reac
         * 			/<sumtotname>           	SumTot
         * 			/<group, pool, reac: recursively>
         * 			/<compartment_name>
         * 				/mesh           		MeshEntry
         * 				/boundary       		Boundary
         * 				/<poolname>         	Pool
         * 				etc, recursively.
    	 * 		/geometry:              		Neutral
    	 * 		/groups:                		Neutral
    	 * 		/graphs:                		Neutral
	 	 * 
		 */
		void makeStandardElements( const Eref& e, const Qinfo*q, string meshClass );
		void meshSplit( const Eref& e, const Qinfo* q, 
			vector< unsigned int > nodeList,
			vector< unsigned int > numEntriesPerNode,
			vector< unsigned int > outgoingEntries,
			vector< unsigned int > incomingEntries
		);
		void meshStats( const Eref& e, const Qinfo* q, 
			unsigned int numMeshEntries, vector< double > voxelVols );

		// Utility functions
		//TreeType findTreeType( const Eref& e );
		Id findChemMesh() const;
		void buildFromBareKineticTree( const string& method );
		void buildFromKkitTree( const Eref& e, const Qinfo* q, const string& method );

		void buildEE( Shell* shell );
		void buildGsl( const Eref& e, const Qinfo* q, Shell* shell, const string& method );
		void buildGssa( const Eref& e, const Qinfo* q, Shell* shell );
		void buildSmoldyn( Shell* shell );

		static const Cinfo* initCinfo();
	private:
		/// syncTime is the interval between synchronizing various solvers.
		double syncTime_;

		/** 
		 * When autoPlot is true, the SimManager builds plots automatically
		 * for fields that it thinks are of interest
		 */
		bool autoPlot_;

		Id baseId_;
		Id stoich_;

		double plotdt_;
		double simdt_;
		double runTime_;
		unsigned int  version_;
		string method_;

		/// Used while partitioning meshed model over many nodes.
		unsigned int numChemNodes_;
};
