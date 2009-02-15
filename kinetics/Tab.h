class Tab: public Data
{
	public:
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );
		void print();
		bool equal( const vector< double >& other, double eps ) const;
		Finfo** initClassInfo();
	private:
		vector< double > vec_;
};
