#include <vector>
#include <string>
#include <sstream>

using namespace std;

vector<string>&
tokenize( const string& line
        , char delimiter
        , vector<string>& words
        )
{
    stringstream ss(line);
    string word;
    while (getline(ss, word, delimiter)) { words.push_back(word); }
    return words;
}
