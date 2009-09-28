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
#if 0
		/**
		 * Converts object into a binary stream. Returns size.
		 */
		virtual unsigned int serialize( vector< char >& buf ) const;

		/**
		 * Creates object from binary stream.
		 */
		virtual Data* unserialize( vector< char >& buf );
#endif
};
