#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include "PyMooseContext.h"
#include "PyMooseBase.h"
#include "Compartment.h"

using namespace std;
extern "C" void __libc_freeres(void);
double clock_overhead()
{
    clock_t start, time, end;
    long count = 0;
    
    time = clock();
    do
    {
        start = clock();
    }
    while (start == time);
    end = start + CLOCKS_PER_SEC;
    while ( (time = clock()) < end)
    {
        ++count;
    }
    return (double)(time - start)/count;
}

    
int main(int argc, char **argv)
{
    atexit(__libc_freeres);
    long loopCount = 10000;
    
    clock_t start, end;
    
    if (argc == 2 )
    {
        loopCount = atol(argv[1]);
    }
    start = clock();
    PyMooseContext* context = PyMooseContext::createPyMooseContext("context", "shell");    
    cerr << "TEST:: PyMooseContext::testPyMooseContext - " <<  (context->testPyMooseContext(loopCount, true)?"SUCCESS":"FAILED") << endl;        
    end = clock();
    cerr << "Time spent to test " << loopCount << " times = " << (double)(end-start)/CLOCKS_PER_SEC << " with overhead = " << clock_overhead() << endl;
//     start = clock();
//     cerr << "TEST:: Compartment::testCompartment() - " << (Compartment::testCompartment(10000,true)?"SUCCESS":"FAILED") << endl;
//     end = clock();
//     cerr << "Time spent to test " << loopCount << " times = " << (double)(end-start)/CLOCKS_PER_SEC << " with overhead = " << clock_overhead() << endl;
    
    return 0;   
}

