class Reac: public Data
{
	friend void testAsync();
		
	public: 
		Reac( double kf, double kb );
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		// void clearQ( Eref e );
		// void doOp( const Conn* c ); //The Conn has src, dest and arg info

		void setKf( double v );
		void setKb( double v );

		Finfo** initClassInfo();
	private:
		double kf_;
		double kb_;
};
