/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Multiscale Object Oriented Simulation Environment.
 **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#include <iostream>
#include <cstdlib>
#include <Python.h>

using namespace std;

int main()
{
	setenv("LD_LIBRARY_PATH", "../cudaLibrary", 1);
	Py_Initialize();
	FILE *file = fopen("TestModel.py", "r+");
	if(file != NULL) {
		PyRun_SimpleFile(file, "TestModel.py");
	}
	fclose(file);
	Py_Finalize();


	return 0;
}

