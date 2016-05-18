# 

import sys
import codecs
import lxml
from lxml import builder
from lxml import etree

#read line-by-line from the input file 
def openFile(inputFilePath):
    fopen = open(str(inputFilePath), 'r')
    content = fopen.readlines()
    newList = []
    final = []
    newatt = ''
    print content
    for att in content:
        newatt = att.replace("\n","")
        newatt = newatt.split("=")
        newList.append(newatt)
    print newList
    for lists in newList:
        for item in lists:
            final.append(item)
    print final
    return final

#create new object as a list
def new(inputFilePath):
    aMap = openFile(inputFilePath)
    return aMap

#convert list to dictionary
def listToDict(aMap):    
    dictionary = dict(zip(aMap[::2], aMap[1::2]))    # sort of hard-coded here
    print dictionary
    print dictionary.items()						 #returns a list
    return dictionary

# the original code to write in XML    
# def writeXMl(aMap):
#     dictionary = listToDict(aMap)
#     ROOT = etree.Element("root")
#     MODEL = etree.SubElement(ROOT, "model")
#     for key, value in dictionary.items():
#         ENTRY = etree.SubElement(MODEL, "entry")
#         ENTRY.text = ("%s = %s" %(key, value))
#     RUNTIME = etree.SubElement(ROOT, "runtime", status= "complete")
#     SIMTIME = etree.SubElement(RUNTIME, "simtime", type = "double", units = "sec")
#     SIMTIME.text = "2000"
#     RESULTS = etree.SubElement(RUNTIME, "runtime", format = "csv")
#     OUTFILE = etree.SubElement(RESULTS, "outfile")
#     OUTFILE.text = "results.csv"
#     runinfo = etree.tostring(RUNTIME, pretty_print = True)
#     doc = etree.tostring(ROOT, pretty_print = True)
#     return doc

#new XML
def writeXMl(aMap):
    dictionary = listToDict(aMap)
    ROOT = etree.Element("root")
    MODEL = etree.SubElement(ROOT, "model")
    META = etree.SubElement(MODEL, "meta")
    META.text = "URL, comments, etc or can be user defined or written during output"
    for key, value in dictionary.items():
    	# valuetype = type(value).__name__
    	print value
    	print type(value)
    	print eval(value)
    	# print type(eval(str(value)))
        PARAMETER = etree.SubElement(MODEL, "parameter", id = str(key), type = value)
        PARAMETER.text = str(value)  #, str(valuetype)
    RUNTIME = etree.SubElement(ROOT, "runtime", status= "complete")
    SIMTIME = etree.SubElement(RUNTIME, "simtime", type = "double", units = "sec")
    SIMTIME.text = "2000"        									#can be changed using time() module
    RESULTS = etree.SubElement(RUNTIME, "runtime", format = "csv")
    OUTFILE = etree.SubElement(RESULTS, "outfile")
    OUTFILE.text = "results.csv"
    runinfo = etree.tostring(RUNTIME, pretty_print = True)
    doc = etree.tostring(ROOT, pretty_print = True, xml_declaration = True, encoding = "UTF-8")
    return doc

#write the final output file
def output(aMap):
    xmldoc = writeXMl(aMap)
    outfile =  codecs.open("./rdesOut.xml",'w','utf-8') 
    sys.stdout = outfile
    print xmldoc
    outfile.close()

# A modified version of the following piece of code(bounded between BELOW and ABOVE)...
# will be used for appending the input xml file
#
#       ----BELOW---- 
#
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



# def getTypeFromString