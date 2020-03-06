# -*- coding: utf-8 -*-
# Script to test all import modules.

import moose

def test_import():
    import moose.genesis 
    import moose.SBML 
    import moose.chemMerge
    import moose.utils 
    import moose.network_utils
    print('done')

def main():
    test_import()

if __name__ == '__main__':
    main()
