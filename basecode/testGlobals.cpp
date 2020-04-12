/***
 *    Description:  Tests functions in global.h
 *
 *        Created:  2020-04-03

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  MIT License
 */

#include "global.h"
#include "../utility/simple_assert.hpp"

void test_normalize_path()
{
    string p1("//a/./b");
    auto p1fixes = moose::normalizePath(p1);
    std::cout << p1 << " " << p1fixes << std::endl;

    string p2("//a/./././///b");
    auto p2fixes = moose::normalizePath(p2);
    std::cout << p2 << " " << p2fixes << std::endl;
    SIMPLE_ASSERT( p2fixes == "/a/b");
}

int main(int argc, const char *argv[])
{
    std::cout << "Testing normalize path" << std::endl;
    test_normalize_path();
    return 0;
}

