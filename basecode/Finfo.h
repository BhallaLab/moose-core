typedef unsigned int (*OpFunc )( Eref e, const void* buf );

class Finfo
{
	public:
		Finfo( OpFunc op );
		unsigned int op( Eref e, const void* buf );
		
		/*
		{
			static_cast< Reac* >( d )->func_( 
				*static_cast< const double* >( buf ) );
			return sizeof( double );
		}
		*/
	private:
		OpFunc op_;
};


/*
unsigned int setKf( Eref e, const char* buf ) {
	static_cast< Reac* >( e.data() )->setKf( *static_cast< const double* >( buf ) );
	return sizeof( double );
}
*/
