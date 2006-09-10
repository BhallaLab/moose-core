#ifndef _OffsetOf_h
#define _OffsetOf_h

#include <cstddef>	// size_t

#ifdef NO_OFFSETOF
#define		FIELD_OFFSET( T, F ) \
	( &T::F )
#else
#define		FIELD_OFFSET( T, F ) \
	( offsetof( T, F ) )
#endif

#endif
