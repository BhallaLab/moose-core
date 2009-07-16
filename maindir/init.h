/*******************************************************************
 * File:            init.h
 * Description:     Functions to do initialization for moose
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 14:52:47
 ********************************************************************/

#ifndef _INIT_H
#define _INIT_H
unsigned int init(int& argc, char**& argv);
void initMPI( int& argc, char**& argv );
void initMoose( int argc, char** argv );
void initParCommunication();
void initSched();
void initParSched();
void initGlobals();
void doneInit();

void pollPostmaster();

#endif
