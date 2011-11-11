# /*******************************************************************
#  * File:             pymoose.py
#  * Description:      This is a wrapper over moose.py and apart from
#  *                   exposing the functions thereof, it adds some 
#  *                   utility functions.
#  * Author1:          Subhasis Ray
#  * E-mail1:          ray dot subhasis at gmail dot com
#  * Created:          2008-10-12 22:50:06
#  * Author2:          Aditya Gilra
#  * E-mail2:          aditya underscore gilra at yahoo dot com
#  ********************************************************************/
# /**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Messaging Object Oriented Simulation Environment,
# ** also known as GENESIS 3 base code.
# **           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
# ** It is made available under the terms of the
# ** GNU General Public License version 2
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
import types
import parser
import token
import symbol
import string
import os
import moose as __moose
from .mooseConstants import *

def listmsg(pymoose_object):
    """Prints the incoming and outgoing messages of the given object."""
    obj = pymoose_object
    ret = []
    if type(pymoose_object) is type(""):
        obj = __moose.Neutral(pymoose_object)
    for msg in obj.inMessages():
        ret.append(msg)
    for msg in obj.outMessages():
        ret.append(msg)
    return ret


def showmsg(pymoose_object):
    """Prints the incoming and outgoing messages of the given object."""
    obj = pymoose_object
    if type(pymoose_object) is type(""):
        obj = __moose.Neutral(pymoose_object)
    print 'INCOMING:'
    for msg in obj.inMessages():
        print msg
    print 'OUTGOING:'
    for msg in obj.outMessages():
        print msg


def readtable(table, filename, separator=None):
    """Reads the file specified by filename to fill the MOOSE table object.

    The file can either have one float on each line, in that case the
    table will be filled with values sequentially.
    Or, the file can have 
    index value
    on each line. If the separator between index and value is anything other than
    white space, you can specify it in the separator argument."""

    in_file = open(filename)
    ii = 0
    line_no = 0
    for line in in_file:
        line_no = line_no + 1
        tokens = split(line, separator)
        if len(token) is 0:
            continue
        elif len(token) == 1:
            table[ii] = float(token[0])
        elif len(token) == 2:
            table[int(token[0])] = float(token[1])
        else:
            print "pymoose.readTable(", table, ",", filename, ",", separator, ") - line#", line_no, " does not fit." 

def getfields(moose_object):
    """Returns a dictionary of the fields (ValueFinfo) in this
    object."""
    fields = {}
    c_dict = moose_object.__class__.__dict__
    for key, value in c_dict.items():
        if type(value) is property:
            try:
                getter = c_dict["_" + moose_object.__class__.__name__ + "__get_" + key]
                fields[key] = getattr(moose_object, key)
            except KeyError:
                pass

    return fields

def apply_to_tree(moose_wildcard, python_filter=None, value=None):
    """
    Select objects by a moose/genesis wildcard, apply a python filter on them and apply a value on them.

    moose_wildcard - this follows GENESIS convention.

    {path}/#[{condition}] returns all elements directly under {path} that satisfy condition. For example:

    '/mynetwork/mycell_0/#[TYPE=Compartment]'

    will return all Compartment objects directly under mycell_0 in mynetwork.

    '{path}/##[{condition}]' will recursively go through all the
    objects that are under {path} (i.e. children, grandchildren,
    great-grandchildren and so on up to the leaf level) and a list of
    the ones meet {condition} will be obtained.

    Thus, '/mynetwork/##[TYPE=Compartment]' will return all
    compartments under mynetwork or its children, or children thereof
    and so on.

    python_filter - if a single string, it will be taken as a
    fieldname, and value will be assigned to this field. It can also
    be a lambda function returning True or False which will be applied
    to each id in the id list returned by moose wildcard
    search. Remember, the argument to the lambda will be an Id, so it
    is up to you to wrap it into a moose object of appropriate type. An example is:

    lambda moose_id: Compartment(moose_id).diameter <  2e-6

    If your moose_wildcard selected objects of Compartment class, then
    this lambda function will select only those with diameter less
    than 2 um.

    value - can be a lambda function to apply arbitrary operations on
    the selected objects.

    If python_filter is a string it, the return
    value of applying the lambda for value() will assigned to the
    field specified by python_filter.

    But if it is value is a data object and {python_filter} is a
    string, then {value} will be assigned to the field named
    {python_filter}.


    If you want to assign Rm = 1e6 for each compartment in mycell
    whose name match 'axon_*':
    
    apply_to_tree('/mycell/##[Class=Compartment]',
            lambda x: 'axon_' in Neutral(x).name,
            lambda x: setattr(Compartment(x), 'Rm', 1e6))

    [you must use setattr to assign value to a field because lambda
    functions don't allow assignments].
    """
    if not isinstance(moose_wildcard, str):
        raise TypeError('moose_wildcard must be a string.')
    id_list = __moose.context.getWildcardList(moose_wildcard, True)
    if isinstance(python_filter, types.LambdaType):
        id_list = [moose_id for moose_id in id_list if python_filter(moose_id)]
    elif isinstance(python_filter, str):
        id_list = [moose_id for moose_id in id_list if hasattr(eval('__moose.%s(moose_id)' % (__moose.Neutral(moose_id).className)), python_filter)]
    else:
        pass
    if isinstance(value, types.LambdaType):
        if isinstance(python_filter, str):
            for moose_id in id_list:
                moose_obj = eval('__moose.%s(moose_id)' % (__moose.Neutral(moose_id).className))
                setattr(moose_obj, python_filter, value(moose_id))
        else:
            for moose_id in id_list:
                value(moose_id)
    else:
        if isinstance(python_filter, str):
            for moose_id in id_list:
                moose_obj = eval('__moose.%s(moose_id)' % (__moose.Neutral(moose_id).className))
                setattr(moose_obj, python_filter, value)
        else:
            raise TypeError('Second argument must be a string specifying a field to assign to when third argument is a value')
            

def tweak_field(moose_wildcard, field, assignment_string):
    """Tweak a specified field of all objects that match the
    moose_wildcard using assignment string. All identifiers in
    assignment string must be fields of the target object.

    Example:

    tweak_field('/mycell/##[Class=Compartment]', 'Rm', '1.5 / (3.1416 * diameter * length')

    will assign Rm to every compartment in mycell such that the
    specific membrane resistance is 1.5 Ohm-m2.
    """    
    if not isinstance(moose_wildcard, str):
        raise TypeError('moose_wildcard must be a string.')
    id_list = __moose.context.getWildcardList(moose_wildcard, True)
    expression = parser.expr(assignment_string)
    expr_list = expression.tolist()
    # This is a hack: I just tried out some possible syntax trees and
    # hand coded the replacement such that any identifier is replaced
    # by moose_obj.identifier
    def replace_fields_with_value(x):
        if len(x)>1:
            if x[0] == symbol.power and x[1][0] == symbol.atom and x[1][1][0] == token.NAME:
                field = x[1][1][1]
                x[1] = [symbol.atom, [token.NAME, 'moose_obj']]
                x.append([symbol.trailer, [token.DOT, '.'], [token.NAME, field]])
            for item in x:
                if isinstance(item, list):
                    replace_fields_with_value(item)
        return x
    tmp = replace_fields_with_value(expr_list)
    new_expr = parser.sequence2st(tmp)
    code = new_expr.compile()
    for moose_id in id_list:
        moose_obj = eval('__moose.%s(moose_id)' % (__moose.Neutral(moose_id).className))
        value = eval(code)
        __moose.context.setField(moose_id, field, str(value))
        
def printtree(root, vchar='|', hchar='__', vcount=1, depth=0, prefix='', is_last=False):
    """Pretty-print a MOOSE tree.
    
    root - the root element of the MOOSE tree, must be some derivatine of Neutral.

    vchar - the character printed to indicate vertical continuation of
    a parent child relationship.

    hchar - the character printed just before the node name

    vcount - determines how many lines will be printed between two
    successive nodes.

    depth - for internal use - should not be explicitly passed.

    prefix - for internal use - should not be explicitly passed.

    is_last - for internal use - should not be explicitly passed.

    """
    if isinstance(root, str) or isinstance(root, __moose.Id):
        root = __moose.Neutral(root)

    for i in range(vcount):
        print prefix

    if depth != 0:
        print prefix + hchar,
        if is_last:
            index = prefix.rfind(vchar)
            prefix = prefix[:index] + ' ' * (len(hchar) + len(vchar)) + vchar
        else:
            prefix = prefix + ' ' * len(hchar) + vchar
    else:
        prefix = prefix + vchar

    print root.name
    
    children = [ __moose.Neutral(child) for child in root.children() ]
    for i in range(0, len(children) - 1):
        printtree(children[i],
                  vchar, hchar, vcount, depth + 1, 
                  prefix, False)
    if len(children) > 0:
        printtree(children[-1], vchar, hchar, vcount, depth + 1, prefix, True)


def df_traverse(root, operation, *args):
    """Traverse the tree in a depth-first manner and apply the
    operation using *args. The first argument is the root object by
    default.""" 
    if hasattr(root, '_visited'):
        return
    operation(root, *args)
    for child in root.children():
    	childNode = __moose.Neutral(child)
        df_traverse(childNode, operation, *args)
    root._visited = True

def readcell_scrambled(filename, target):
    """A special version for handling cases where a .p file has a line
    with specified parent yet to be defined.

    It creates a temporary file with a sorted version based on
    connectivity, so that parent is always defined before child."""
    pfile = open(filename, "r")
    tmpfilename = filename + ".tmp"
    graph = defaultdict(list)
    data = {}
    error = None
    root = None
    for line in pfile:
        tmpline = line.strip()
        if tmpline.startswith("*") or tmpline.startswith("//"):
            continue
        elif tmpline.startswith("/*"):
            error = "Handling C style comments not implemented."
            break
        node, parent, rest, = tmpline.split(None, 2)
        print node, parent
        if (parent == "none"):
            if (root is None):
                root = node
            else:
                error = "Duplicate root elements: ", root, node, "> Cannot process any further."
                break
        graph[parent].append(node)
        data[node] = line
    if error is not None:
        print error
        return None

    tmpfile = open(tmpfilename, "w")
    stack = [root]
    while stack:
        current = stack.pop()
        children = graph[current]
        stack.extend(children)
        tmpfile.write(data[current])
    tmpfile.close()
    __moose.context.readCell(tmpfilename, target)
    return __moose.Cell(target)

############# added by Aditya Gilra -- begin ################

def resetSim(context, simdt, plotdt):
    """ Sets the simdt and plotdt and resets the MOOSE 'context'. """
    context.setClock(0, simdt, 0)
    context.setClock(1, simdt, 0) #### The hsolve and ee methods use clock 1
    context.setClock(2, simdt, 0) #### hsolve uses clock 2 for mg_block, nmdachan and others.
    context.setClock(PLOTCLOCK, plotdt, 0) # PLOTCLOCK is in mooseConstants.py
    context.reset()

def setupTable(name, obj, qtyname):
    """ Sets up a table with 'name' which stores 'qtyname' from 'obj'. """
    # Setup the tables to pull data
    vmTable = __moose.Table(name, __moose.Neutral(obj.path+"/data"))
    vmTable.stepMode = TAB_BUF #TAB_BUF: table acts as a buffer.
    vmTable.connect("inputRequest", obj, qtyname)
    vmTable.useClock(PLOTCLOCK)
    return vmTable

def connectSynapse(context, compartment, synname, gbar_factor):
    """
    Creates a synname synapse under compartment, sets Gbar*gbar_factor, and attaches to compartment.
    synname must be a synapse in /library of MOOSE.
    """
    synapseid = context.deepCopy(context.pathToId('/library/'+synname),\
        context.pathToId(compartment.path),synname)
    synapse = __moose.SynChan(synapseid)
    synapse.Gbar = synapse.Gbar*gbar_factor
    if synapse.getField('mgblock')=='True': # If NMDA synapse based on mgblock, connect to mgblock
        mgblock = __moose.Mg_block(synapse.path+'/mgblock')
        compartment.connect("channel", mgblock, "channel")
    else:
        compartment.connect("channel", synapse, "channel")
    return synapse

def connect_CaConc(compartment_list):
    """ Connect the Ca pools and channels within each of the compartments in compartment_list
     Ca channels should have an extra field called 'ion' defined and set in MOOSE.
     Ca dependent channels like KCa should have an extra field called 'ionDependency' defined and set in MOOSE.
     Call this only after instantiating cell so that all channels and pools have been created. """
    context = __moose.PyMooseBase.getContext()
    for compartment in compartment_list:
        caconc = None
        for child in compartment.getChildren(compartment.id):
            neutralwrap = __moose.Neutral(child)
            if neutralwrap.className == 'CaConc':
                caconc = __moose.CaConc(child)
                break
        if caconc is not None:
            for child in compartment.getChildren(compartment.id):
                neutralwrap = __moose.Neutral(child)
                if neutralwrap.className == 'HHChannel':
                    channel = __moose.HHChannel(child)
                    ## If 'ion' field is not present, the Shell returns '0',
                    ## cribs and prints out a message but it does not throw an exception
                    if channel.getField('ion') in ['Ca','ca']:
                        channel.connect('IkSrc',caconc,'current')
                        #print 'Connected ',channel.path
                if neutralwrap.className == 'HHChannel2D':
                    channel = __moose.HHChannel2D(child)
                    ## If 'ionDependency' field is not present, the Shell returns '0',
                    ## cribs and prints out a message but it does not throw an exception
                    if channel.getField('ionDependency') in ['Ca','ca']:
                        caconc.connect('concSrc',channel,'concen')
                        #print 'Connected ',channel.path

def printNetTree():
    """ Prints all the cells under /, and recursive prints the cell tree for each cell. """
    root = __moose.Neutral('/')
    for id in root.getChildren(root.id): # all subelements of 'root'
        if __moose.Neutral(id).className == 'Cell':
            cell = __moose.Cell(id)
            print "-------------------- CELL : ",cell.name," ---------------------------"
            printCellTree(cell)

def printCellTree(cell):
    """
    Prints the tree under MOOSE object 'cell'.
    Assumes cells have all their compartments one level below,
    also there should be nothing other than compartments on level below.
    Apart from compartment properties and messages,
    it displays the same for subelements of compartments only one level below the compartments.
    Thus NMDA synapses' mgblock-s will be left out.
    """
    for compartmentid in cell.getChildren(cell.id): # compartments
        comp = __moose.Compartment(compartmentid)
        print "  |-",comp.path, 'l=',comp.length, 'd=',comp.diameter, 'Rm=',comp.Rm, 'Ra=',comp.Ra, 'Cm=',comp.Cm, 'EM=',comp.Em
        for inmsg in comp.inMessages():
            print "    |---", inmsg
        for outmsg in comp.outMessages():
            print "    |---", outmsg
        printRecursiveTree(compartmentid, level=2) # for channels and synapses and recursively lower levels

def printRecursiveTree(elementid, level):
    """ Recursive helper function for printCellTree,
    specify depth/'level' to recurse and print subelements under MOOSE 'elementid'. """
    spacefill = '  '*level
    element = __moose.Neutral(elementid)
    for childid in element.getChildren(elementid): 
        childobj = __moose.Neutral(childid)
        classname = childobj.className
        if classname in ['SynChan','KinSynChan']:
            childobj = __moose.SynChan(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'Gbar=',childobj.Gbar
        elif classname in ['HHChannel', 'HHChannel2D']:
            childobj = __moose.HHChannel(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'Gbar=',childobj.Gbar, 'Ek=',childobj.Ek
        elif classname in ['CaConc']:
            childobj = __moose.CaConc(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'thick=',childobj.thick, 'B=',childobj.B
        elif classname in ['Mg_block']:
            childobj = __moose.Mg_block(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'CMg',childobj.CMg, 'KMg_A',childobj.KMg_A, 'KMg_B',childobj.KMg_B
        elif classname in ['Table']: # Table gives segfault if printRecursiveTree is called on it
            return # so go no deeper
        for inmsg in childobj.inMessages():
            print spacefill+"  |---", inmsg
        for outmsg in childobj.outMessages():
            print spacefill+"  |---", outmsg
        if len(childobj.getChildren(childid))>0:
            printRecursiveTree(childid, level+1)

def setup_vclamp(compartment, name, delay1, width1, level1, gain=0.5e-5):
    """
    Sets up a voltage clamp with 'name' on MOOSE 'compartment' object:
    adapted from squid.g in DEMOS (moose/genesis)
    Specify the 'delay1', 'width1' and 'level1' of the voltage to be applied to the compartment.
    Typically you need to adjust the PID 'gain'
    For perhaps the Davison 4-compartment mitral or the Davison granule:
    0.5e-5 optimal gain - too high 0.5e-4 drives it to oscillate at high frequency,
    too low 0.5e-6 makes it have an initial overshoot (due to Na channels?)
    Returns a MOOSE table with the PID output.
    """
    ## If /elec doesn't exists it creates /elec and returns a reference to it.
    ## If it does, it just returns its reference.
    __moose.Neutral('/elec')
    pulsegen = __moose.PulseGen('/elec/pulsegen'+name)
    vclamp = __moose.DiffAmp('/elec/vclamp'+name)
    vclamp.saturation = 999.0
    vclamp.gain = 1.0
    lowpass = __moose.RC('/elec/lowpass'+name)
    lowpass.R = 1.0
    lowpass.C = 50e-6 # 50 microseconds tau
    PID = __moose.PIDController('/elec/PID'+name)
    PID.gain = gain
    PID.tau_i = 20e-6
    PID.tau_d = 5e-6
    PID.saturation = 999.0
    # All connections should be written as source.connect('',destination,'')
    pulsegen.connect('outputSrc',lowpass,'injectMsg')
    lowpass.connect('outputSrc',vclamp,'plusDest')
    vclamp.connect('outputSrc',PID,'commandDest')
    PID.connect('outputSrc',compartment,'injectMsg')
    compartment.connect('VmSrc',PID,'sensedDest')
    
    pulsegen.trigMode = 0 # free run
    pulsegen.baseLevel = -70e-3
    pulsegen.firstDelay = delay1
    pulsegen.firstWidth = width1
    pulsegen.firstLevel = level1
    pulsegen.secondDelay = 1e6
    pulsegen.secondLevel = -70e-3
    pulsegen.secondWidth = 0.0

    vclamp_I = __moose.Table("/elec/vClampITable"+name)
    vclamp_I.stepMode = TAB_BUF #TAB_BUF: table acts as a buffer.
    vclamp_I.connect("inputRequest", PID, "output")
    vclamp_I.useClock(PLOTCLOCK)
    
    return vclamp_I

def setup_iclamp(compartment, name, delay1, width1, level1):
    """
    Sets up a current clamp with 'name' on MOOSE 'compartment' object:
    Specify the 'delay1', 'width1' and 'level1' of the current pulse to be applied to the compartment.
    Returns the MOOSE pulsegen that sends the current pulse.
    """
    ## If /elec doesn't exists it creates /elec and returns a reference to it.
    ## If it does, it just returns its reference.
    __moose.Neutral('/elec')
    pulsegen = __moose.PulseGen('/elec/pulsegen'+name)
    iclamp = __moose.DiffAmp('/elec/iclamp'+name)
    iclamp.saturation = 1e6
    iclamp.gain = 1.0
    pulsegen.trigMode = 0 # free run
    pulsegen.baseLevel = 0.0
    pulsegen.firstDelay = delay1
    pulsegen.firstWidth = width1
    pulsegen.firstLevel = level1
    pulsegen.secondDelay = 1e6 # to avoid repeat
    pulsegen.secondLevel = 0.0
    pulsegen.secondWidth = 0.0
    pulsegen.connect('outputSrc',iclamp,'plusDest')
    iclamp.connect('outputSrc',compartment,'injectMsg')
    return pulsegen

def get_matching_children(parent, names):
    """ Returns non-recursive children of 'parent' MOOSE object
    with their names containing any of the strings in list 'names'. """
    matchlist = []
    for childID in parent.children():
        child = __moose.Neutral(childID)
        for name in names:
            if name in child.name:
                matchlist.append(childID)
    return matchlist

def underscorize(path):
    """ Returns: / replaced by underscores in 'path' """
    return string.join(string.split(path,'/'),'_')

def blockChannels(cell, channel_list):
    """
    Sets gmax to zero for channels of the 'cell' specified in 'channel_list'
    Substring matches in channel_list are allowed
    e.g. 'K' should block all K channels (ensure that you don't use capital K elsewhere in your channel name!)
    """
    for compartmentid in cell.getChildren(cell.id): # compartments
        comp = __moose.Compartment(compartmentid)
        for childid in comp.getChildren(comp.id):
            child = __moose.Neutral(childid)
            if child.className in ['HHChannel', 'HHChannel2D']:
                chan = __moose.HHChannel(childid)
                for channame in channel_list:
                    if channame in chan.name:
                        chan.Gbar = 0.0

############# added by Aditya Gilra -- end ################

if __name__ == "__main__": # test printtree
    s = __moose.Neutral('cell')
    soma = __moose.Neutral('soma', s)
    d1 = __moose.Neutral('d1', soma)
    d2 = __moose.Neutral('d2', soma)
    d3 = __moose.Neutral('d3', d1)
    d4 = __moose.Neutral('d4', d1)
    d5 = __moose.Neutral('d5', s)
    printtree(s)

    expected = 'cell            \
                |               \
                |__soma         \
                |  |            \
                |  |__d1        \
                |  |  |         \
                |  |  |__d3     \
                |  |  |         \
                |  |  |__d4     \
                |  |            \
                |  |__d2        \
                |               \
                |__d5'

    s1 = __moose.Neutral('cell1')
    c1 = __moose.Neutral('c1', s1)
    c2 = __moose.Neutral('c2', c1)
    c3 = __moose.Neutral('c3', c1)
    c4 = __moose.Neutral('c4', c2)
    c5 = __moose.Neutral('c5', c3)
    c6 = __moose.Neutral('c6', c3)
    c7 = __moose.Neutral('c7', c4)
    c8 = __moose.Neutral('c8', c5)
    printtree(s1)
    expected1 = 'cell1                  \
                 |                      \
                 |__c1                  \
                    |                   \
                    |__c2               \
                    |  |                \
                    |  |__c4            \
                    |     |             \
                    |     |__c7         \
                    |                   \
                    |__c3               \
                       |                \
                       |__c5            \
                       |  |             \
                       |  |__c8         \
                       |                \
                       |__c6'
