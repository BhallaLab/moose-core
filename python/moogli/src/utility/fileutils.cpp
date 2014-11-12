#include <fstream>
#include <string>
#include <streambuf>

using namespace std;

unsigned int
file_get_contents( const string& filename
                 ,       string& buffer
                 )
{
    ifstream file(filename.c_str());
    file.seekg(0, ios::end);
    buffer.reserve(file.tellg());
    file.seekg(0, ios::beg);
    buffer.assign( (istreambuf_iterator<char>(file))
                 , istreambuf_iterator<char>()
                 );
    file.close();
    return buffer.size();
}

bool
file_exists(const string& filename)
{
    ifstream file(filename.c_str(), ifstream::in);
    if(!file) { return false; }
    file.close();
    return true;
}
