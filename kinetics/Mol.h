class Mol: public Data
{
	friend void testSyncArray( unsigned int size, unsigned int numThreads,
		unsigned int method );
	friend void checkVal( double time, const Mol* m, unsigned int size );
	friend void forceCheckVal( double time, Element* e, unsigned int size );

	public: 
		Mol()
			: nInit_( 0.0 ), A_( 0.0 ), B_( 0.0 )
			{;}

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
