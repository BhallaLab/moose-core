/*
 * =====================================================================================
 *
 *       Filename:  cnpy.h
 *
 *    Description:  Write a stl vector to numpy format 2.
 *
 *      This program is part of MOOSE simulator.
 *
 *        Version:  1.0
 *        Created:  05/04/2016 10:36:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#ifndef  cnpy_INC
#define  cnpy_INC

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <complex>
#include <typeinfo>

#include <memory>
#include <array>
#include <string>
#include <stdint.h>

#include "../utility/print_function.hpp"


using namespace std;

namespace cnpy2
{

// Check the endian-ness of machine at run-time. This is from library
// https://github.com/rogersce/cnpy
char BigEndianTest();

void split(vector<string>& strs, string& input, const string& pat);

/**
 * @brief Check if a numpy file is sane or not.
 *
 * Read first 8 bytes and compare with standard header.
 *
 * @param npy_file Path to file.
 *
 * @return  true if file is sane, else false.
 */
bool isValidNumpyFile(std::ifstream& fp);

/**
 * @brief Parser header from a numpy file. Store it in vector.
 *
 * @param header
 */
void findHeader(std::fstream& fp, string& header );

/**
 * @brief Change shape in numpy header.
 *
 * @param
 * @param data_len
 * @param
 */
void changeHeaderShape(std::ofstream& fs, const size_t data_len, const size_t numcols);

// Use version 2.0 of npy fommat.
// https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
static vector<char> __pre__ {
    (char)0x93, 'N', 'U', 'M', 'P', 'Y'         /* Magic */
    , (char)0x02, (char) 0x00               /* format */
};

size_t writeHeader(std::fstream& fp, const vector<string>& colnames, const vector<size_t>& shape);

void writeNumpy(const string& outfile, const vector<double>& vec, const vector<string>& colnames);

void appendNumpy(const string& outfile, const vector<double>& vec, const vector<string>& colnames);

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  initialize a numpy file with given column names.
 *
 * @Param filename
 * @Param colnames
 */

size_t initNumpyFile(const string& outfile, const vector<string>& colnames);


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  read numpy file and return data in a vector. This for testing
 * purpose and should not used to read data.
 *
 * @Param infile
 * @Param data
 */
/* ----------------------------------------------------------------------------*/
void readNumpy(const string& infile, vector<double>& data);

} // Namespace cnpy2 ends.

#endif   /* ----- #ifndef cnpy_INC  ----- */
