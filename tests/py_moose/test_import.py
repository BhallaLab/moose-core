# -*- coding: utf-8 -*-
# Script to test all import modules.

import moose

def test_import():
    import moose.genesis 
    import moose.SBML 
    import moose.chemMerge
    import moose.utils as mu
    print('done')

def main():
    test_import()

if __name__ == '__main__':
    main()
