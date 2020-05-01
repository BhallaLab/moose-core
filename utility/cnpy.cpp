/*
 * =====================================================================================
 *
 *       Filename:  cnpy.cpp
 *
 *    Description:  Write vector to numpy file.
 *
 *        Version:  1.0
 *        Created:  05/05/2016 11:58:40 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#include "cnpy.hpp"
#include <fstream>
#include <iterator>

#include "print_function.hpp"

using namespace std;

namespace cnpy2
{

// Check the endian-ness of machine at run-time. This is from library
// https://github.com/rogersce/cnpy
char BigEndianTest()
{
    unsigned char x[] = {1,0};
    short y = *(short*) x;
    return y == 1 ? '<' : '>';
}

size_t headerSize = 0;

void split(vector<string>& strs, string& input, const string& pat)
{
    char* pch;
    pch = strtok( &input[0], pat.c_str() );
    while( pch != NULL )
    {
        strs.push_back( string(pch ) );
        pch = strtok( NULL, pat.c_str() );
    }
    delete pch;
}

string shapeToString(const vector<size_t>& shape)
{
    string s{"("};
    if(! shape.empty())
    {
        s += std::to_string(shape[0]);
        for(size_t i = 1; i < shape.size(); i++)
        {
            s += ",";
            s += std::to_string(shape[i]);
        }
        if(shape.size() == 1)
            s += ",";
    }
    else
        s += "0,";
    s += ")";
    return s;
}

/**
 * @brief Check if a numpy file is sane or not.
 *
 * Read first 8 bytes and compare with standard header.
 *
 * @param npy_file Path to file.
 *
 * @return  true if file is sane, else false.
 */
bool isValidNumpyFile( FILE* fp )
{
    assert( fp );
    char buffer[__pre__.size()];
    size_t nr = fread( buffer, sizeof(char), __pre__.size(), fp );

    if( 0 == nr )
        return false;

    bool equal = true;
    // Check for equality
    for(size_t i = 0; i < __pre__.size(); i++ )
        if( buffer[i] != __pre__[i] )
        {
            equal = false;
            break;
        }
    return equal;
}

/**
 * @brief Parser header from a numpy file. Store it in vector.
 *
 * @param header
 */
void findHeader(std::fstream& fs, string& header )
{
    // Read header, till we hit newline character.
    char ch;
    fs.seekg(0);                                 // Go to the begining.
    header.clear();
    while(! fs.eof())
    {
        fs.get(ch);
        if( '\n' == ch )
            break;
        header.push_back(ch);
    }
    assert( header.size() >= __pre__.size() );
}

/**
 * @brief Change shape in numpy header.
 *
 * @param
 * @param data_len
 * @param
 */
void changeHeaderShape(std::fstream& fs, const size_t data_len, const size_t numcols)
{
    string header{""};

    // Find header. Search for newline.
    findHeader(fs, header);
    const size_t headerSize = header.size();

    size_t shapePos = header.find( "'shape':" );
    size_t lbrac = header.find( '(', shapePos );
    size_t rbrac = header.find( ')', lbrac );
    assert( lbrac > shapePos );
    assert( rbrac > lbrac );

    string prefixHeader = header.substr( 0, lbrac + 1 );
    string postfixHeader = header.substr( rbrac );

    string shapeStr = header.substr( lbrac+1, rbrac-lbrac-1);

    vector<string> tokens;
    split( tokens, shapeStr, "," );

    string newShape = "";
    for (size_t i = 0; i < tokens.size(); i++)
        newShape += std::to_string( atoi( tokens[i].c_str() ) + data_len/numcols ) + ",";

    string newHeader = prefixHeader + newShape + postfixHeader;
    if( newHeader.size() < header.size() )
    {
        cout << "Warn: Modified header can not be smaller than old header" << endl;
    }

    // Resize to the old header size. Newline is not included in the header.
    newHeader.resize(header.size());
    newHeader += '\n';                          // Add newline before writing. 
    fs.seekp(0);
    fs.write(newHeader.c_str(), newHeader.size());
}

size_t writeHeader(std::fstream& fs, const vector<string>& colnames, const vector<size_t>& shape)
{
    // Heder are always at the begining of file.
    fs.seekp(0);

    // Write the format string. 8 bytes.
    fs.write(&__pre__[0], __pre__.size());

    char endianChar = cnpy2::BigEndianTest();
    const char formatChar = 'd';

    // Next 4 bytes are header length. This is computed again when data is
    // appended to the file. We can have maximum of 2^32 bytes of header which
    // ~4GB.

    string header = ""; // This is the header to numpy file
    header += "{'descr':[";
    for( auto it = colnames.cbegin(); it != colnames.end(); it++ )
        header += "('" + *it + "','" + endianChar + formatChar + "'),";

    // shape is changed everytime we append the data. We use fixed number of
    // character in shape. Its a int, we will use 13 chars to represent shape.
    header += "], 'fortran_order':False,'shape':";
    header += shapeToString(shape);
    header += ",}";

    // Add some extra sapce for safety.
    header += string(12, ' ');

    // FROM THE DOC: It is terminated by a newline (\n) and padded with spaces
    // (\x20) to make the total of len(magic string) + 2 + len(length) +
    // HEADER_LEN be evenly divisible by 64 for alignment purposes.
    // pad with spaces so that preamble+headerlen+header is modulo 16 bytes.
    // preamble is 8 bytes, header len is 4 bytes, total 12.
    // header needs to end with \n
    unsigned int remainder = 16 - (12 + header.size()) % 16;
    header.insert(header.end(), remainder-1, ' ');
    header += '\n';                             // Add newline. 

    // Now write the size of header. Its 4 byte long in version 2.
    uint32_t s = header.size();
    fs.write((char*)&s, 4);
    fs << header;
    return fs.tellp();
}


size_t initNumpyFile(const string& outfile, const vector<string>& colnames)
{
    std::fstream fs;
    fs.open(outfile, std::fstream::in | std::fstream::out | std::fstream::trunc | std::ofstream::binary);

    if(! fs.is_open())
    {
        cerr << "Error: Could not create " << outfile << endl;
        return 0;
    }
    vector<size_t> shape;
    auto pos = writeHeader(fs, colnames, shape);
    fs.close();
    return pos;
}

void writeNumpy(const string& outfile, const vector<double>& data, const vector<string>& colnames)
{

    // In our application, we need to write a vector as matrix. We do not
    // support the stacking of matrices.
    vector<size_t> shape;

    if( colnames.size() == 0)
        return;

    shape.push_back(data.size() / colnames.size());

    // Create a new file.
    std::fstream fs;
    fs.open(outfile, std::ofstream::in | std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    /* In mode "w", open the file and write a header as well. When file is open
     * in mode "a", we assume that file is alreay a valid numpy file.
     */
    if(! fs.is_open())
    {
        moose::showWarn( "Could not open file " + outfile );
        return;
    }

    auto p = writeHeader( fs, colnames, shape );
    fs.seekp(p);

    // Here the previous character is '\n'

    fs.write(reinterpret_cast<const char*>(data.data()), sizeof(double)*data.size());
    fs.close();
}

void appendNumpy(const string& outfile, const vector<double>& vec, const vector<string>& colnames)
{

    std::fstream fs;
    fs.open(outfile, std::fstream::in | std::fstream::out | std::fstream::binary);

    if( ! fs.is_open() )
    {
        moose::showWarn( "Could not open " + outfile + " to write " );
        return;
    }

    // And change the shape in header.
    changeHeaderShape(fs, vec.size(), colnames.size());

    // Go to the end.
    fs.seekp(0, std::ios_base::end);
    fs.write(reinterpret_cast<const char*>(&vec[0]), sizeof(double)*vec.size());
    fs.close();
}

void readNumpy(const string& infile, vector<double>& data)
{
    cout << "Reading from " << infile << endl;
    std::ifstream fs;
    fs.open(infile, std::ios::in | std::ios::binary);

    if(! fs.is_open())
    {
        cerr << "Could not open " << infile << endl;
        return;
    }

    char ch;
    fs.get(ch);
    size_t nBytes = 1;
    while(ch != '\n')
    {
        fs.get(ch);
        nBytes += 1;
    }

    char buff[sizeof(double)];
    double x;

    fs.seekg(nBytes, std::ios_base::beg);
    while(! fs.eof())
    {
        fs.read(buff, sizeof(double));
        if(fs.gcount() != 8) 
            break;
        memcpy(&x, buff, sizeof(double));
        data.push_back(x);
    }
    cout << endl;
    fs.close();
}

}                                               /* Namespace cnpy2 ends. */
