"""visualizer.py: KKTI visualiser.

KKIT can be found at: https://www.ncbs.res.in/node/359

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import warnings
from collections import defaultdict
import re
try:
    import networkx as nx
except:
    warnings.warn("Module networkx is not found. You can't use this module")


def short_label(path, len = -1):
    label = path.split('/')[-1]
    if len > -1:
        return label[:len]
    else:
        return label

class KKIT():

    def __init__(self, filepath):
        self.modelFilePath = filepath
        self.lines = None
        self.simobjsDict = {}
        self.objs = defaultdict(list)
        self.edges = {}
        self.nodes = {}
        self.dot = {}
        self.dot = defaultdict(list)
        self.dot['header'].append('graph kkit { \n')
        self.dot['footer'].append('}')
        self.G = nx.DiGraph()

    def add_simobjdump(self, words):
        self.simobjsDict[words[0]] = words[1:]

    def simundump(self, words):
        name = words[0]
        if name not in self.simobjsDict:
            print("simobjdump: %s not declared " % name)
            print("+ Avaialable are: %s" % '\n\t'.join(self.simobjsDict.keys()))
        #if len(words[1:]) != len(self.simobjsDict[name]):
            #print("[WARN] Mismatch in parameters")
            #print("+ %s" % zip(self.simobjsDict[name], words))
            #print("+ {}={}".format(' '.join(self.simobjsDict[name]), ' '.join(words)))
            #pass
        self.objs[name].append(words[1:])
        self.nodes[words[1]] = words
        #self.G.add_node(words[1])

    def split_words(self, line):
        words = line.split('"')
        return line.split()

    def create_graph(self):
        """Create a graph """
        self.G.add_edges_from(self.edges)
        for n in self.G.nodes():
            self.G.node[n]['label'] = short_label(n)
            nType = self.nodes[n][0]
            params = self.nodes[n][1:]
            shape = nType
            if nType == 'kpool': shape = 'egg'
            elif nType == 'kenz': shape = 'Mcircle'
            elif nType == 'kreac':
                shape = 'rect'
                self.G.node[n]['label'] += "\nkf=%s,kb=%s" % (params[2], params[3])
            elif  nType == 'xplot': shape = 'record'
            self.G.node[n]['type'] = nType
            self.G.node[n]['shape'] = shape
        dotFile = "{}.dot".format(self.modelFilePath)
        print("[INFO] Writing dot file to %s" % dotFile)
        nx.write_dot(self.G, dotFile)

        
    def add_line(self, line):
        quotex = re.compile(r'\"[\S\s]+\"', re.I)
        words = self.split_words(line)
        if 'simobjdump' == words[0]:
            self.add_simobjdump(words[1:])
        elif 'simundump' == words[0]:
            self.simundump(words[1:])
        elif 'addmsg' == words[0]:
            A, B = words[1], words[2]
            edgeType = words[3]
            if edgeType.upper() == "SUBSTRATE":
                btype = self.nodes[B][0]
                if btype == "kreac":
                    self.edges[(A, B)] = words[3:]
                else:
                    self.edges[(B, A)] = words[3:]
            elif edgeType.upper() == "REAC":
                if words[4:].index('B') == 0:
                    self.edges[(A, B)] = words[3:]
            elif edgeType.upper() == "ENZYME":
                self.edges[(B, A)] = words[3:]
            elif edgeType.upper() == "MM_PRD":
                self.edges[(B, A)] = words[3:]
            else:
                warnings.warn("msgtype %s not supported" % edgeType)
        else:
            warnings.warn("%s not supported yet" % words[0])

    def build_model(self):
        '''Build kkit model '''
        [self.add_line(l) for l in self.lines]
        self.create_graph()

def join_text(text, line_sep):
    lines = text.split('\n')
    newLines = []
    fullLine = ''
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        if '//' == line[0:2]:
            continue

        if '\\' == line[-1]:
            fullLine += line[:-1]
            continue
        else:
            fullLine += line
            newLines.append(fullLine)
            fullLine = ''
    return newLines

def visualize(kkitFileName):
    '''Parse the kkit file and build model '''
    kkit = KKIT(kkitFileName)
    with open(kkitFileName, "r") as f:
        kkit.lines = join_text(f.read(), line_sep='\\')
    kkit.build_model()

def main():
    kkitmodel = sys.argv[1]
    visualize(kkitmodel)

if __name__ == '__main__':
    main()
