class Data
{
	public:
		virtual ~Data()
			{;}
		virtual void process( const ProcInfo* p, Eref e ) = 0;
		virtual void reinit( Eref e ) = 0;

		/**
		 * Handles incoming synaptic messages. Many objects don't need
		 * it, so we don't insist.
		 */
		virtual void addSpike( unsigned int synId, double time )
		{
			;
		}

		/**
		 * Every Data class must provide a function to initialize its
		 * ClassInfo.
		 */
		virtual Finfo** initClassInfo() = 0;
};
