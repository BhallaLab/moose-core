class ProcInfo
{
	public:
		ProcInfo() 
			: dt( 1.0 ), currTime( 0.0 ), numThreads( 1 )
			{;}
		double dt;
		double currTime;
		unsigned int numThreads;
};

typedef ProcInfo* ProcPtr;
