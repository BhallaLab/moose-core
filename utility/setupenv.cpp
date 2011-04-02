// setupenv.cpp --- 
// 
// Filename: setupenv.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Mar 26 22:36:10 2011 (+0530)
// Version: 
// Last-Updated: Mon Mar 28 15:50:58 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 17
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 
// 

// Code:

#include <map>
#include <string>
#include <sstream>
#include <cstdlib>
using namespace std;

extern unsigned getNumCores();

const map<string, string>& getArgMap()
{
    static map<string, string> argmap;
    if (argmap.empty()){
        char * isSingleThreaded = getenv("SINGLETHREADED");
        if (isSingleThreaded != NULL){
            argmap.insert(pair<string, string>("SINGLETHREADED", string(isSingleThreaded)));
        }
        else {
            argmap.insert(pair<string, string>("SINGLETHREADED", "0"));
        }
        char * isInfinite = getenv("INFINITE");
        if (isInfinite != NULL){
         argmap.insert(pair<string, string>("INFINITE", string(isInfinite)));
        }
        else {
            argmap.insert(pair<string, string>("INFINITE", "0"));
        }   
        char * numCores = getenv("NUMCORES");
        if (numCores != NULL){
            argmap.insert(pair<string, string>("NUMCORES", string(numCores)));
        } else {
            unsigned int cores = getNumCores();
            stringstream s;
            s << cores;
            argmap.insert(pair<string, string>("NUMCORES", s.str()));        
        }
        char * numNodes = getenv("NUMNODES");
        if (numNodes != NULL){
            argmap.insert(pair<string, string>("NUMNODES", string(numNodes)));
        } else {
            argmap.insert(pair<string, string>("NUMNODES", "1"));
        }
    }
    return argmap;
}



// 
// setupenv.cpp ends here
