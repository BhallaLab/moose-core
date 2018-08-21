# -*- coding: utf-8 -*-
## Wrapper function to run all the tests here in batch mode.

def main():
    for i in [ "testXreacs2", "testXreacs3", "testXreacs4a", "testXreacs4",
    "testXreacs5a", "testXreacs5", "testXreacs6", "testXreacs7",
    "testXreacs8", "testXchan1", "testXdiff1", "testXenz1",]:
        print(i)
        j = __import__( i )
        j.main()

if __name__ == '__main__':
    main()


