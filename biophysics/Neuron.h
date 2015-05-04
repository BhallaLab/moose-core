/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURON_H
#define _NEURON_H

/**
 * The Neuron class to hold the Compartment class elements.
 */

class Neuron
{
	public:
		Neuron();
		void setRM( double v );
		double getRM() const;
		void setRA( double v );
		double getRA() const;
		void setCM( double v );
		double getCM() const;
		void setEm( double v );
		double getEm() const;
		void setTheta( double v );
		double getTheta() const;
		void setPhi( double v );
		double getPhi() const;
		void setSourceFile( string v );
		string getSourceFile() const;
		void setCompartmentLengthInLambdas( double v );
		double getCompartmentLengthInLambdas() const;
		unsigned int getNumCompartments() const;
		unsigned int getNumBranches() const;
		vector< double> getPathDistFromSoma() const;
		vector< double> getGeomDistFromSoma() const;
		vector< double> getElecDistFromSoma() const;
		vector< ObjId > getCompartments() const;
		void setChannelDistribution( vector< string > v );
		vector< string > getChannelDistribution() const;
		void setMechSpec( vector< string > v );
		vector< string > getMechSpec() const;
		void setSpineSpecification( vector< string > v );
		vector< string > getSpineSpecification() const;

		void buildSegmentTree( const Eref& e );

		///////////////////////////////////////////////////////////////////
		// MechSpec set
		///////////////////////////////////////////////////////////////////
		void updateSegmentLengths();
		void installSpines( const vector< ObjId >& elist,
			const vector< double >& val, const vector< string >& line );
		void makeSpacingDistrib( 
			const vector< ObjId >& elist, const vector< double >& val,
			vector< unsigned int >& elistIndex, vector< double >& pos,
			const vector< string >& line ) const;
		void parseMechSpec( const Eref& e );
		void installMechanism(  const string& name,
			const vector< ObjId >& elist, const vector< double >& val,
			const vector< string >& line );
		void evalExprForElist( const vector< ObjId >& elist,
			const string& expn, vector< double >& val );

		///////////////////////////////////////////////////////////////////
		// Old set
		///////////////////////////////////////////////////////////////////
		void makeSpacingDistrib( vector< double >& pos,
			double spacing, double spacingDistrib );
		void insertSpines( const Eref& e, Id spineProto, string path,
			vector< double > placement );
		void parseSpines( const Eref& e );
		void clearSpines( const Eref& e );

		void assignChanDistrib( const Eref& e,
			string name, string path, string func );
		void clearChanDistrib( const Eref& e,
			string name, string path );
		void parseChanDistrib( const Eref& e );
		void evalChanParams( const string& name, const string& func, 
						vector< ObjId >& elist );
		

		/**
		 * Initializes the class info.
		 */
		static const Cinfo* initCinfo();
	private:
		double RM_;
		double RA_;
		double CM_;
		double Em_;
		double theta_;
		double phi_;
		Id soma_;
		string sourceFile_;
		double compartmentLengthInLambdas_;
		unsigned int spineIndex_;
		vector< string > channelDistribution_;
		vector< string > spineSpecification_;
		vector< string > mechSpec_;

		/// Map to look up Seg index from Id of associated compt.
		map< Id, unsigned int > segIndex_; 
		vector< Id > segId_; /// Id of each Seg entry, below.
		vector< SwcSegment > segs_;
		vector< SwcBranch > branches_;

};

// 

#endif // 
