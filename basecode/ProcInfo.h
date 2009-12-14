class ProcInfo
{
	public:
		ProcInfo() 
			: dt( 1.0 ), currTime( 0.0 ), numThreads( 1 ), threadId( 0 )
			{;}
		double dt;
		double currTime;
		unsigned int numThreads;
		unsigned int threadId;
		unsigned int node;
};

typedef ProcInfo* ProcPtr;
