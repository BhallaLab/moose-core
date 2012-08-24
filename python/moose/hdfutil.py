# hdfutil.py --- 
# 
# Filename: hdfutil.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Aug 23 17:34:55 2012 (+0530)
# Version: 
# Last-Updated: Fri Aug 24 15:44:04 2012 (+0530)
#           By: subha
#     Update #: 155
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Utility function to save data in hdf5 format
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import moose as moose__
import numpy as np
import h5py as h5
import time

cpptonp = {
    'int': 'i8',
    'bool': 'b',
    'unsigned int': 'u8',
    'float': 'f8',
    'string': 'S1024'}
    
size_step=256
def savestate(filename=None):
    """Dump the state of MOOSE in an hdf5 file.
    
    The file will have a data set for each class.
    Each such dataset will be a column of field values.
    """    
    if filename is None:
        filename = 'moose_session_' + time.strftime('%Y%m%d_%H%M%S') + '.hdf5'
    with h5.File(filename, 'w') as fd:
        root = fd.create_group('/root')
        class_dataset_dict = {}
        class_count_dict = {}
        class_array_dict = {}
        for obj in moose__.wildcardFind("/##"):
            if obj.path.startswith('/Msg') or obj.path.startswith('/class') or obj.class_ == 'Table' or obj.class_ == 'TableEntry':
                continue
            # print obj.path, obj.class_
            if obj.class_ not in class_dataset_dict:
                # print 'Creating entries for class:', obj.class_
                fielddict = moose__.getFieldDict(obj.class_, 'valueFinfo')
                keys = sorted(list(fielddict.keys()))
                fields = [] # [('path', 'S1024')]
                for fieldname in keys:
                    ftype = fielddict[fieldname]
                    if ftype in cpptonp:
                        fields.append((fieldname, cpptonp[ftype]))
                    elif ftype == 'Id' or ftype == 'ObjId':
                        fields.append((fieldname, 'S1024'))
                # print fields
                ds = root.create_dataset(obj.class_, shape=(size_step,), dtype=fields, compression='gzip', compression_opts=6)
                class_dataset_dict[obj.class_] = ds
                class_array_dict[obj.class_] = []
                class_count_dict[obj.class_] = 0
            ds = class_dataset_dict[obj.class_]
            for entry in obj:
                fields = [entry.getField(f) for f in ds.dtype.names]
                fields = [f.path if isinstance(f, moose__.ematrix) or isinstance(f, moose__.element) else f for f in fields]
                class_array_dict[obj.class_].append(fields)
                # print 'fields:'
                # print fields
                # print 'length:', len(class_array_dict[obj.class_])
                class_count_dict[obj.class_] += 1
                if class_count_dict[obj.class_] == size_step:
                    oldlen = ds.len()
                    if oldlen <= class_count_dict[obj.class_]:
                        ds.resize((class_count_dict[obj.class_]))
                    ds[oldlen: class_count_dict[obj.class_]] = np.rec.array(class_array_dict[obj.class_], dtype=ds.dtype)
                    class_array_dict[obj.class_] = []
        for classname in class_array_dict:
            ds = class_dataset_dict[classname]
            ds.resize((class_count_dict[classname], ))
            if len(class_array_dict[classname]) > 0:
                start = class_count_dict[classname] - len(class_array_dict[classname])
                ds[start:] = np.rec.array(class_array_dict[classname], dtype=ds.dtype)
            
                        

# 
# hdfutil.py ends here
