class Data
{
	public:
		virtual ~Data()
			{;}
		virtual void process( const ProcInfo* p, Eref e ) = 0;
		virtual void reinit( Eref e ) = 0;
};
