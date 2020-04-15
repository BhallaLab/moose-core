# -*- coding: utf-8 -*-
import moose
import re

def fixPath(path):
    path = re.sub(r'/+', '/', path)
    return path

def test_path():
    paths = [ '/a'
            , '//a'
            , '/a/b'
            , '/a/b/'
            , '//a//b/////'
            , '/a/./b'
            , '///a/././b'
            ]

    expectedPath = set([fixPath(p) for p in paths])
    print(expectedPath)

    for p in paths:
        print("-- Creating %s" % p)
        if not moose.exists(p):
            print('exists')
            p = moose.Neutral(p)
        else:
            p = moose.element(p)
        print(p)

    foundPath = []
    for p in moose.wildcardFind('/##'):
        if "/a" in p.path:
           foundPath.append(p.path)

    testFailed = False
    for f in foundPath:
        f = re.sub(r'\[\d+\]', '', f)
        if f not in expectedPath:
            testFailed = True

    assert not testFailed, "Test failed"


def main():
    test_path()

if __name__ == '__main__':
    main()
