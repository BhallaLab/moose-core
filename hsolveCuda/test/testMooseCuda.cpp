/*
 * testMoose.cpp
 *
 *  Created on: Jun 4, 2013
 *      Author: saeed
 */

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

