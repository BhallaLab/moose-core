# ninemlio.py --- 
# 
# Filename: ninemlio.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Tue May 31 11:24:53 2011 (+0530)
# Version: 
# Last-Updated: Tue May 31 11:41:24 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 22
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 2011-05-31 11:25:07 (+0530) Initial version
# 

# Code:

import nineml.user_layer as ninemlul

def read_model(filename):    
    with open(filename) as model_file:
        model = ninemlul.parse(model_file)
        print dir(model)
        model.check()
        # These are preliminery investigations, will be superseeded by
        # actual object creation in moose.
        for name, component in model.components.items():
            print 'Component:', name, component
        for name, group in model.groups.items():
            print 'Group:',  name, group

def test_read_model():
    read_model('simple_example.xml')
    print 'Successfully parsed simple_example.xml'

if __name__ == '__main__':
    test_read_model()
    print '9ml test main finished.'
# 
# ninemlio.py ends here
