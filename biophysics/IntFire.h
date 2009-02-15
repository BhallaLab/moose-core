class IntFire: public Data
{
	public: 
		IntFire( double thresh, double tau );
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		// void generalQ( Eref e );
		//
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		void setVm( double v );
		void setTau( double v );
		void setThresh( double v );
		Finfo** initClassInfo();
	private:
		double Vm_; // State variable: Membrane potential. Resting pot is 0.
		double thresh_;	// Firing threshold
		double tau_; // Time course of membrane settling.
		vector< double > w_;
		double X_; // state variable
		double Y_; // state variable
};
