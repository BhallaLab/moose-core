#ifndef _MARKOVCHANNEL_H
#define _MARKOVCHANNEL_H

//This class deals with ion channels which can be found in one of multiple 
//states, some of which are conducting. This implementation assumes the 
//occurence of first order kinetics to calculate the probabilities of the
//channel of being found in all states. Further, the rates of transition 
//between these states can be constant, voltage-dependent, ligand dependent 
//(only one ligand species) or both. The current flow obtained from the channel
//is calculated in a deterministic method by solving the system of
//differential equations obtained from the assumptions above.*/

class MarkovChannel : public ChanBase {
	public:
	//Default constructor. Use is not recommended as most of the class members
	//cannot be initialized. 
	MarkovChannel();

	//Constructor to be used when number of states and number of open states are
	//known. Use of this constructor is recommended as all the other members of
	//the class can be initialized with the information provided. 
	static const Cinfo* initCinfo();

	MarkovChannel( unsigned int, unsigned int);
	~MarkovChannel( );

	unsigned getNumStates( ) const;
	void setNumStates( unsigned int );

	unsigned int getNumOpenStates( ) const;
	void setNumOpenStates( unsigned int );

	vector< string > getStateLabels( ) const;
	void setStateLabels( vector< string > );

	//If the (i,j)'th is true, ligand concentration will be used instead of
	//voltage.
	vector< vector< bool > > getLigandGated( ) const;
	void setLigandGated ( vector< vector< bool > > );

	//Probabilities of the channel occupying all possible states.
	vector< double > getState ( ) const;
	void setState(  vector< double >  );

	//The initial state of the channel. State of the channel is reset to this
	//vector during a call to reinit().
	vector< double > getInitialState() const;
	void setInitialState( vector< double > );

	//Conductances associated with each open/conducting state.
	vector< double > getGbars( ) const;
	void setGbars( vector< double > );

	//Couple of dummy get functions. 
	vector< double > getOneParamRateTable( unsigned int, unsigned int );
	vector< vector< double > > getTwoParamRateTable( unsigned int, unsigned int );


	//Type-independent lookup function for rate.
	double lookupRate( unsigned int, unsigned int, vector<double> );

	//Updating the rates of transiton at each time step.
	void updateRates();

	//Sets those rates of transition which are constant throughout the time
	//interval.
	void initConstantRates();

/*	//Function to return status of initialization of channel. Returns true if all
	//the parameters of the channel i.e. number of states, number of open states,
	//conductances, rate tables have all been initialized. This is to prevent
	//accidental changes in parameters while setting parameters.
	bool channelIsInitialized();	*/

	//GSL related functions.
	static int evalGslSystem( double t, const double* y, double* yprime, void* s );
	int innerEvalGslSystem( double t, const double* y, double* yprime );

	//DestFinfo functions.
	void setupRateTables( unsigned int );
	void setOneParamRateTable( vector< unsigned int >, vector< double >,  vector< double >, bool ligandFlag );
	void setTwoParamRateTable( vector< unsigned int >, vector< double >, vector< vector< double > >);
	void process( const Eref&, const ProcPtr);
	void reinit( const Eref&, const ProcPtr);
	void handleLigandConc( double );

	private:
	double g_;												//Expected conductance of the channel.	
	double ligandConc_;								//Ligand concentration.
	unsigned int numStates_;					//Total number of states.
	unsigned int numOpenStates_;			//Number of open (conducting) states.
	vector< vector< double > > A_;		//Instantaneous rate matrix.

	vector< string > stateLabels_;
	vector< double > state_;					//Probabilities of occupancy of each state.
	vector< double > initialState_;
	vector< double > Gbars_;		//Conductance associated with each open state.
//	bool isInitialized_;

	MarkovRateTable* rateTables_;

	double* stateForGsl_;
	MarkovGsl* solver_;
};

#endif
