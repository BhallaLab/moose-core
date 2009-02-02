class IntFire: public Data
{
	public: 
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		void generalQ( Eref e );
	private:
		double v_;
		double Em_;
		double tau_;
		double vThresh_;
		double Gm_; // Membrane conductance
		vector< double > w_;
};
