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
	testAsync();

	cout << endl;
	return 0;
}

