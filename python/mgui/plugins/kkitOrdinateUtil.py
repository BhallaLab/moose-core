from moose import *
import numpy as np
#import matplotlib.pyplot as plt
#import pygraphviz as pgv
import networkx as nx
from collections import Counter
#import matplotlib.pyplot as plt

def xyPosition(objInfo,xory):
    try:
        return(float(element(objInfo).getField(xory)))
    except ValueError:
        return (float(0))

def setupMeshObj(modelRoot):
    ''' Setup compartment and its members pool,reaction,enz cplx under self.meshEntry dictionaries \ 
    self.meshEntry with "key" as compartment, 
    value is key2:list where key2 represents moose object type,list of objects of a perticular type
    e.g self.meshEntry[meshEnt] = { 'reaction': reaction_list,'enzyme':enzyme_list,'pool':poollist,'cplx': cplxlist }
    '''
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    meshEntry = {}
    xcord = []
    ycord = []
    meshEntryWildcard = '/##[ISA=ChemCompt]'
    if modelRoot != '/':
        meshEntryWildcard = modelRoot+meshEntryWildcard

    for meshEnt in wildcardFind(meshEntryWildcard):
        print meshEnt
        mollist = []
        realist = []
        enzlist = []
        cplxlist = []
        tablist = []

        mol_cpl = wildcardFind(meshEnt.path+'/##[ISA=PoolBase]')
        enzlist = wildcardFind(meshEnt.path+'/##[ISA=EnzBase]')
        realist = wildcardFind(meshEnt.path+'/##[ISA=ReacBase]')
        tablist = wildcardFind(meshEnt.path+'/##[ISA=StimulusTable]')
    
        for m in mol_cpl:
            if isinstance(element(m.parent),CplxEnzBase):
                cplxlist.append(m)
                objInfo = m.parent.path+'/info'
            elif isinstance(element(m),moose.PoolBase):
                mollist.append(m)
                objInfo =m.path+'/info'
            xcord.append(xyPosition(objInfo,'x'))
            ycord.append(xyPosition(objInfo,'y')) 

        getxyCord(xcord,ycord,enzlist)
        getxyCord(xcord,ycord,realist)
        getxyCord(xcord,ycord,tablist)

        meshEntry[meshEnt] = {'enzyme':enzlist,
                              'reaction':realist,
                              'pool':mollist,
                              'cplx':cplxlist,
                              'table':tablist
                              }
        xmin = min(xcord)
        xmax = max(xcord)
        ymin = min(ycord)
        ymax = max(ycord)
        noPositionInfo = len(np.nonzero(xcord)[0]) == 0 \
            and len(np.nonzero(ycord)[0]) == 0
    return(meshEntry,xmin,xmax,ymin,ymax,noPositionInfo)

def getxyCord(xcord,ycord,list):
    for item in list:
        objInfo = item.path+'/info'
        xcord.append(xyPosition(objInfo,'x'))
        ycord.append(xyPosition(objInfo,'y'))

def setupItem(modelPath,cntDict):
    '''This function collects information of what is connected to what. \
    eg. substrate and product connectivity to reaction's and enzyme's \
    sumtotal connectivity to its pool are collected '''

    zombieType = ['ReacBase','EnzBase','FuncBase','StimulusTable']
    for baseObj in zombieType:
        path = '/##[ISA='+baseObj+']'
        if modelPath != '/':
            path = modelPath+path
        if ( (baseObj == 'ReacBase') or (baseObj == 'EnzBase')):
            for items in wildcardFind(path):
                sublist = []
                prdlist = []
                uniqItem,countuniqItem = countitems(items,'subOut')
                for sub in uniqItem: 
                    sublist.append((sub,'s',countuniqItem[sub]))

                uniqItem,countuniqItem = countitems(items,'prd')
                for prd in uniqItem:
                    prdlist.append((prd,'p',countuniqItem[prd]))
                
                if (baseObj == 'CplxEnzBase') :
                    uniqItem,countuniqItem = countitems(items,'toEnz')
                    for enzpar in uniqItem:
                        sublist.append((enzpar,'t',countuniqItem[enzpar]))
                    
                    uniqItem,countuniqItem = countitems(items,'cplxDest')
                    for cplx in uniqItem:
                        prdlist.append((cplx,'cplx',countuniqItem[cplx]))

                if (baseObj == 'EnzBase'):
                    uniqItem,countuniqItem = countitems(items,'enzDest')
                    for enzpar in uniqItem:
                        sublist.append((enzpar,'t',countuniqItem[enzpar]))
                cntDict[items] = sublist,prdlist
        elif baseObj == 'FuncBase':
            #ZombieSumFunc adding inputs
            for items in wildcardFind(path):
                inputlist = []
                outputlist = []
                funplist = []
                nfunplist = []
                uniqItem,countuniqItem = countitems(items,'input')
                for inpt in uniqItem:
                    inputlist.append((inpt,'st',countuniqItem[inpt]))
                for funcbase in moose.element(items).neighbors['output']: 
                    funplist.append(funcbase)
                if(len(funplist) > 1): print "SumFunPool has multiple Funpool"
                else:  cntDict[funplist[0]] = inputlist
        else:
            for tab in wildcardFind(path):
                tablist = []
                uniqItem,countuniqItem = countitems(tab,'output')
                for tabconnect in uniqItem:
                    tablist.append((tabconnect,'tab',countuniqItem[tabconnect]))
                cntDict[tab] = tablist
def countitems(mitems,objtype):
    items = []
    items = element(mitems).neighbors[objtype]
    uniqItems = set(items)
    countuniqItems = Counter(items)
    return(uniqItems,countuniqItems)

def autoCoordinates(meshEntry,srcdesConnection):
    #for cmpt,memb in meshEntry.items():
    #    print memb
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    G = nx.Graph()
    for cmpt,memb in meshEntry.items():
        for enzObj in find_index(memb,'enzyme'):
            G.add_node(enzObj.path)
    for cmpt,memb in meshEntry.items():
        for poolObj in find_index(memb,'pool'):
            G.add_node(poolObj.path)
        for cplxObj in find_index(memb,'cplx'):
            G.add_node(cplxObj.path)
            G.add_edge((cplxObj.parent).path,cplxObj.path)
        for reaObj in find_index(memb,'reaction'):
            G.add_node(reaObj.path)
        
    for inn,out in srcdesConnection.items():
        if (inn.className =='ZombieReac'): arrowcolor = 'green'
        elif(inn.className =='ZombieEnz'): arrowcolor = 'red'
        else: arrowcolor = 'blue'
        if isinstance(out,tuple):
            if len(out[0])== 0:
                print inn.className + ':' +inn[0].name + "  doesn't have input message"
            else:
                for items in (items for items in out[0] ):
                    G.add_edge(element(items[0]).path,inn.path)
            if len(out[1]) == 0:
                print inn.className + ':' + inn[0].name + "doesn't have output mssg"
            else:
                for items in (items for items in out[1] ):
                    G.add_edge(inn.path,element(items[0]).path)
        elif isinstance(out,list):
            if len(out) == 0:
                print "Func pool doesn't have sumtotal"
            else:
                for items in (items for items in out ):
                    G.add_edge(element(items[0]).path,inn.path)
    
    nx.draw(G,pos=nx.spring_layout(G))
    #plt.savefig('/home/harsha/Desktop/netwrokXtest.png')
    xcord = []
    ycord = []
    position = nx.spring_layout(G)
    for y in position.values():
        xcord.append(y[0])
        ycord.append(y[1])
	    
    return(min(xcord),max(xcord),min(ycord),max(ycord),position)

def find_index(value, key):
    """ Value.get(key) to avoid expection which would raise if empty value in dictionary for a given key """
    if value.get(key) != None:
        return value.get(key)
    else:
        raise ValueError('no dict with the key found')
