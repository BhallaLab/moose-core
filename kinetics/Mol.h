class Mol: public Data
{
	public: 
		Mol( double nInit )
			: nInit_( nInit ), A_( 0.0 ), B_( 0.0 )
			{;}

		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		// void clearQ( Eref e );
		// void doOp( const Conn* c ); //The Conn has src, dest and arg info
		Finfo** initClassInfo();
	private:
		double n_;
		double nInit_;
		double A_;
		double B_;
};
