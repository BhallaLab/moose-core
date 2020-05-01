/*******************************************************************
 * File:            StringUtil.cpp
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 12:12:10
 ********************************************************************/

#include <string>
#include <sstream>
#include <regex>
#include <cassert>
#include <iostream>
#include <vector>

#include "strutil.h"

using namespace std;

namespace moose {

// Adapted from code available on oopweb.com
void tokenize(const string& str, const string& delimiters,
              vector<string>& tokens)
{
    // Token boundaries
    string::size_type begin = str.find_first_not_of(delimiters, 0);
    string::size_type end = str.find_first_of(delimiters, begin);

    while(string::npos != begin || string::npos != end) {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(begin, end - begin));

        // Update boundaries
        begin = str.find_first_not_of(delimiters, end);
        end = str.find_first_of(delimiters, begin);
    }
}

string& clean_type_name(string& arg)
{
    for(size_t pos = arg.find(' '); pos != string::npos; pos = arg.find(' ')) {
        arg.replace(pos, 1, 1, '_');
    }
    for(size_t pos = arg.find('<'); pos != string::npos; pos = arg.find('<')) {
        arg.replace(pos, 1, 1, '_');
    }
    for(size_t pos = arg.find('>'); pos != string::npos; pos = arg.find('>')) {
        arg.replace(pos, 1, 1, '_');
    }
    return arg;
}

std::string trim(const std::string myString, const string& delimiters)
{
    if(myString.length() == 0) {
        return myString;
    }

    string::size_type end = myString.find_last_not_of(delimiters);
    string::size_type begin = myString.find_first_not_of(delimiters);

    if(begin != string::npos) {
        return std::string(myString, begin, end - begin + 1);
    }

    return "";
}

std::string fix(const std::string userPath, const string& delimiters)
{
    string trimmedPath = trim(userPath, delimiters);

    string fixedPath;
    char prev = 0;

    // In this loop, we check if there are more than one '/' together. If yes,
    // then accept only first one and reject other.
    for(unsigned int i = 0; i < trimmedPath.size(); ++i) {
        const char c = trimmedPath[i];
        if(c != '/' || c != prev)
            fixedPath.push_back(c);
        prev = c;
    }
    return fixedPath;
}

int testTrim()
{

    std::string testStrings[] = {" space at beginning",
                                 "space at end ",
                                 " space at both sides ",
                                 "\ttab at beginning",
                                 "tab at end\t",
                                 "\ttab at both sides\t",
                                 "\nnewline at beginning",
                                 "newline at end\n",
                                 "\nnewline at both sides\n",
                                 "\n\tnewline and tab at beginning",
                                 "space and tab at end \t",
                                 "   \rtab and return at both sides \r"};

    std::string results[] = {"space at beginning",
                             "space at end",
                             "space at both sides",
                             "tab at beginning",
                             "tab at end",
                             "tab at both sides",
                             "newline at beginning",
                             "newline at end",
                             "newline at both sides",
                             "newline and tab at beginning",
                             "space and tab at end",
                             "tab and return at both sides"};

    bool success = true;

    for(unsigned int i = 0; i < sizeof(testStrings) / sizeof(*testStrings);
        ++i) {
        std::string trimmed = trim(testStrings[i]);
        success = (results[i].compare(trimmed) == 0);

        cout << "'" << trimmed << "'" << (success ? " SUCCESS" : "FAILED")
             << endl;
    }
    return success ? 1 : 0;
}

bool endswith(const string& full, const string& ending)
{
    if(full.length() < ending.length()) {
        return false;
    }
    return (0 == full.compare(full.length() - ending.length(), ending.length(),
                              ending));
}

/* Compare two strings. */
int strncasecmp(const string& a, const string& b, size_t n)
{
    for(size_t i = 0; i < std::min(n, b.size()); ++i)
        if(tolower(a[i]) != tolower(b[i]))
            return tolower(a[i]) - tolower(b[i]);

    if(b.size() < n)
        return a.size() - b.size();

    return 0;
}

// This function is modification of this solution:
// https://stackoverflow.com/a/440240/1805129
string random_string(const unsigned len)
{
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    string s(len, '_');
    for(unsigned i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return s;
}

void str_replace_all(string& str, const string& a, const string& b)
{
    if(a.size() == 0)
        return;

    size_t index = 0;
    while(true) {
        /* Locate the substring to replace. */
        index = str.find(a, index);

        if(index == std::string::npos)
            break;

        /* Make the replacement. */
        str.erase(index, a.size());
        str.insert(index, b);
    }
}

bool isPrefix(const string& a, const string& b)
{
    if(a.size() < b.size())
        return false;
    return (b.find(a, 0) == 0);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Split a given path into (parent, name). The behaviour of this
 * function is akin to unix'; dirname and basepath.
 *
 * @Param path
 *
 * @Returns A pair of strings <parent, name>
 */
/* ----------------------------------------------------------------------------*/
std::pair<std::string, std::string> splitPath(const std::string& path)
{
    assert(path[0] == '/');
    auto i = path.find_last_of('/');
    auto parentPath = i > 0 ? path.substr(0, i) : "/";
    return std::make_pair(parentPath, path.substr(i + 1));
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Normalize a given path by removing multiple repeating // to / and
 * /./ to /
 *
 * @Param path
 *
 * @Returns
 */
/* ----------------------------------------------------------------------------*/
string normalizePath(const string& path)
{
    string s(path);
    static std::regex e0("/+");  // Remove multiple / by single /
    s = std::regex_replace(s, e0, "/");
    static std::regex e1("/(\\./)+");  // Remove multiple / by single /
    s = std::regex_replace(s, e1, "/");
    return s;
}

void split(const string& text, const string& delimiter, vector<string>& res)
{
    // From https://stackoverflow.com/a/46931770/1805129
    string token;
    size_t pos_start = 0, pos_end, delim_len = delimiter.size();
    while ((pos_end = text.find (delimiter, pos_start)) != string::npos) {
        token = text.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }
    res.push_back (text.substr (pos_start));
}

string textwrap(const string& text, const string& prefix, const size_t width)
{
    vector<string> words;
    tokenize(text, " \n", words);
    string res;
    size_t size = 0;
    for(const auto w : words) {
        if(size == 0) {
            res += prefix;
            size = prefix.size();
        }
        res += w + ' ';
        size += w.size() + 1;
        if(size > width) {
            res += '\n';
            size = 0;
        }
    }
    return res;
}

std::string boxed(const string& text, const size_t width)
{
    return fmt::format(
        "┌{0:─^{2}}┐\n"
        "│{1: ^{2}}│\n"
        "└{0:─^{2}}┘\n",
        "", text, width);
}

std::string capitalize(const string& f)
{
    string ff(f);
    ff[0] = std::toupper(ff[0]);
    return ff;
}

}  // namespace moose
