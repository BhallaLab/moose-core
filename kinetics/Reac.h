class Reac: public Data
{
	public: 
		Reac( double kf, double kb );
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		// void clearQ( Eref e );
		// void doOp( const Conn* c ); //The Conn has src, dest and arg info
	private:
		double kf_;
		double kb_;
};
