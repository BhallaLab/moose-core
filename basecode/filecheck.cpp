/*******************************************************************
 * File:            filecheck.cpp
 * Description:     Implements utility functions to open files
 *                  with error checking
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-04-22 02:33:58
 ********************************************************************/
#ifndef _FILECHECK_CPP
#define _FILECHECK_CPP
#include "filecheck.h"
using namespace std;

bool file_exists(std::string fileName)
{
    ifstream fin;
    fin.open(fileName.c_str());
    if ( fin.fail())
    {
        return false;
    }
    fin.close();
    return true;    
}

bool open_outfile(std::string fileName, ofstream& outfile)
{
    if ( file_exists(fileName))
    {
        return false;
    }
    outfile.clear();
    outfile.open(fileName.c_str());
    return !outfile.fail();
}

bool open_infile (std::string filename, ifstream & fin) {
  fin.clear();  // Clears any error flags from previous fail() calls
  fin.open (filename.c_str());
  return !fin.fail();
}

bool open_appendfile(std::string fileName, ofstream& outfile)
{
    outfile.clear();
    outfile.open(fileName.c_str(),ios::app);
    return !outfile.fail();
}
#endif
