template< class T, class A, void ( T::*F )( A ) >
	unsigned int async1( Eref e, const void* buf )
{
	(static_cast< T* >( e.data() )->*F)( 
		*static_cast< const A* >( buf ) );
	return sizeof( FuncId ) + sizeof ( A );
}
