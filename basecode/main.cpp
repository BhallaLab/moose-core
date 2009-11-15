#include "header.h"
#include "Shell.h"
#include <sys/time.h>
#include <math.h>
#include <queue>

const FuncId ENDFUNC( -1 );

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testScheduling();

void init()
{
	Id shellid = Shell::initCinfo()->create( "root", 1 );
	assert ( shellid == Id() );
	SetGet::setShell();
}

int main()
{
	init();
#ifdef DO_UNIT_TESTS
	cout << "testing: ";
	testAsync();
	testScheduling();
#endif
	cout << endl;

	delete Id()();
	return 0;
}

