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

void init()
{
	Id shellid = Shell::initCinfo()->create( "root", 1 );
	assert ( shellid == Id() );
	SetGet::setShell();
}

int main()
{
	init();
	cout << "testing: ";
	testAsync();

	cout << endl;

	delete Id()();
	return 0;
}

