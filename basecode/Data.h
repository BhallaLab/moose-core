class Data
{
	public:
		virtual ~Data()
			{;}
		virtual void process( const ProcInfo* p, Eref e ) = 0;

		/**
		 * Every Data class must provide a function to initialize its
		 * ClassInfo.
		 */
		virtual Finfo** initClassInfo() = 0;
};
