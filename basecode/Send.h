template < class T > void send1( Eref e, Slot src, T val )
{
	double* buf = e.getBufPtr( src );
	*static_cast< T* >( buf ) = val;
}

template < class T1, class T2 > void send2( Eref e, Slot src, T1 v1, T2 v2 )
{
	double* buf = e.getBufPtr( src );
	*static_cast< T1* >( buf ) = v1;
	buf += sizeof( T1 ) / sizeof( double );
	*static_cast< T2* >( buf ) = v2;
}
