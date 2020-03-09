// =====================================================================================
//
//       Filename:  test_cnpy.cpp
//
//    Description: Test cnpy. 
//
//        Version:  1.0
//        Created:  Monday 09 March 2020 01:22:51  IST
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Dilawar Singh (), dilawar.s.rajput@gmail.com
//   Organization:  NCBS Bangalore
//
// =====================================================================================

#include "cnpy.hpp"

#include <vector>
#include <cstdlib>

using namespace std;


int main(int argc, const char *argv[])
{
    srand(time(NULL));

    string datafile = "_a_data.npy";
    vector<string> cols {"A", "B", "C", "P", "Q" };

    // Now append data.
    vector<double> data;
    for (size_t i = 0; i < cols.size(); i++) 
        for (size_t ii = 0; ii < 10; ii++) 
            data.push_back(rand() / (double)RAND_MAX);

    cout << "Size of data is " << data.size() << endl;

    cnpy2::writeNumpy(datafile, data, cols);
    vector<double> r1;
    cnpy2::readNumpy(datafile, r1);
    cout << "Size of data without append: " << r1.size() << endl;
    for (size_t i = 0; i < r1.size(); i++)
        assert(r1[i] == data[i]);


    datafile = "_b_data.npy";
    cnpy2::initNumpyFile(datafile, cols);
    cnpy2::appendNumpy(datafile, data, cols);
    cnpy2::appendNumpy(datafile, data, cols);
    cnpy2::appendNumpy(datafile, data, cols);

    vector<double> r2;
    cnpy2::readNumpy(datafile, r2);

    cout << "Total data read " << r2.size() << endl;
    assert(r2.size() == 3 * data.size() );

    for (size_t i = 0; i < r2.size(); i++)
        assert(data[i%data.size()] == r2[i]);
    
    return 0;
}
