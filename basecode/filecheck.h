/*******************************************************************
 * File:            filecheck.h
 * Description:     Declares utility functions for opening files
 *                  with check for prior existence and errors.
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-04-11 12:28:00
 ********************************************************************/
#ifndef _FILECHECK_H
#define _FILECHECK_H
#include <string>
#include <iostream>
#include <fstream>
/**
  These are helper function for file opening and closing
*/
bool file_exists(std::string fileName);
bool open_outfile(std::string fileName, std::ofstream& outfile);
bool open_infile (std::string filename, std::ifstream & fin);
bool open_appendfile(std::string fileName, std::ofstream& outfile);
#endif
