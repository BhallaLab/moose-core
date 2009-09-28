#include "header.h"
#include <sys/time.h>
#include <math.h>
#include <queue>

const FuncId ENDFUNC( -1 );

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );

int main()
{
	cout << "testing: ";
//	testSync();
	testAsync();

	// Test single thread
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 1, 0 );

	// Test pthreads barrier
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 2, 0 );
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 4, 0 );

	// Test myBarrier
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 2, 1 );
	for ( unsigned int size = 10; size < 10001; size *= 10 )
		testSyncArray( size, 4, 1 );

	cout << endl;
	return 0;
}

