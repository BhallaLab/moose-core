#include "header.h"

Finfo::Finfo( OpFunc op )
	: op_( op )
{
	;
}

unsigned int Finfo::op( Eref e, const void* buf )
{
	return op_( e, buf );
}
