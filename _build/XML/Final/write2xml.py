#

import sys
from lxml import builder
from lxml import etree
import lxml

#read line-by-line from the input file and return a list 
def openFile(inputFilePath):
    fopen = open(str(inputFilePath), 'r')
    content = fopen.readlines()
    newList = []
    final = []
    newatt = ''
    for att in content:
        newatt = att.replace("\n","")
        newatt = newatt.split("=")
        newList.append(newatt)
    for lists in newList:
        for item in lists:
            final.append(item)
    return final

#create new object
def new(inputFilePath):
    aMap = openFile(inputFilePath)
    return aMap

#convert list to dictionary
def listToDict(aMap):    
    dictionary = dict(zip(aMap[::2], aMap[1::2]))
    return dictionary

#write in XML    
def writeXMl(aMap):
    dictionary = listToDict(aMap)
    ROOT = etree.Element("root")
    MODEL = etree.SubElement(ROOT, "model")
    META = etree.SubElement(MODEL, "meta")
    META.text = "URL, comments, etc or can be user defined or written during output.\n"
    for key, value in dictionary.items():
        PARAMETER = etree.SubElement(MODEL, "parameter", id = str(key))
        PARAMETER.text = str(value)  
    RUNTIME = etree.SubElement(ROOT, "runtime", status= "complete")
    SIMTIME = etree.SubElement(RUNTIME, "simtime", type = "double", units = "sec")
    SIMTIME.text = "2000"                                           #can be changed using time() module
    RESULTS = etree.SubElement(RUNTIME, "runtime", format = "csv")
    OUTFILE = etree.SubElement(RESULTS, "outfile")
    OUTFILE.text = "results.csv"
    runinfo = etree.tostring(RUNTIME, pretty_print = True)
    doc = etree.tostring(ROOT, pretty_print = True, xml_declaration = True, encoding = "UTF-8")
    return doc

#write the final output file
def output(aMap):
    xmldoc = writeXMl(aMap)
    outfile =  open("./rdesOut.xml",'w') 
    sys.stdout = outfile
    print xmldoc
    outfile.close()

# A modified version of the following piece of code(bounded between BELOW and ABOVE)...
# will be used for appending the input xml file
#
#       ----BELOW---- 
# def hash_key(aMap, key):
#     return hash(key)%len(aMap)

# def get_bucket(aMap, key):
#     bucket_id = hash_key(aMap, key)
#     return aMap[bucket_id]

# def get_slot(aMap, key, default=None):
#     bucket = get_bucket(aMap, key)
#     for i, kv in enumerate(bucket):
#         k, v = kv
#         if key == k:
#             return i, k, v

#     return -1, key, default

# def get(aMap, key, default=None):
#     i, k, v = get_slot(aMap, key, default=default)
#     return v

# def set(aMap, key, value):
#     bucket = get_bucket(aMap, key)
#     i, k, v = get_slot(aMap, key)

#     if i >=0:
#         bucket[i] = (key, value)
#     else:
#         bucket.append((key,value))

# def delete(aMap, key):
#     bucket = get_bucket(aMap, key)

#     for i in xrange(len(bucket)):
#         k, v = bucket[i]
#         if key == k:
#             del bucket[i]
#             break

# def lister(aMap):
#     for bucket in aMap:
#         if bucket:
#             for k, v in bucket:
#                 print k, v
#
#        ----ABOVE----

