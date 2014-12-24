#ifndef _STRINGUTILS_HPP_
#define _STRINGUTILS_HPP_

#include <vector>
#include <string>
#include <sstream>

using namespace std;

vector<string>&
tokenize( const string& line
        , char delimiter
        , vector<string>& words
        );

#endif /* _STRINGUTILS_HPP_ */
