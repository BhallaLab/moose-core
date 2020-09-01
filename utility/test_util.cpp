/***
 *    Description:  Test utility functions.
 *
 *        Created:  2020-04-14

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 */

#include <iostream>

#include "utility.h"
#include "strutil.h"
#include "testing_macros.hpp"

using namespace std;

void test_normalize_path()
{
    string p1("//a/./b");
    auto p1fixes = moose::normalizePath(p1);
    std::cout << p1 << " " << p1fixes << std::endl;

    string p2("//a/./././///b");
    auto p2fixes = moose::normalizePath(p2);
    std::cout << p2 << " " << p2fixes << std::endl;
    ASSERT_EQ( p2fixes, "/a/b", "PATH");
}

int main(int argc, const char *argv[])
{
    test_normalize_path();
    return 0;
}

