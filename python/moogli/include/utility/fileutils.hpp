#ifndef _FILEUTILS_HPP_
#define _FILEUTILS_HPP_

#include <string>

using namespace std;

unsigned int
file_get_contents( const string& filename
                 ,       string& buffer
                 );

bool
file_exists(const string& filename);

#endif /* _FILEUTILS_HPP_ */
