import string, os
import moose
from mooseConstants import *
from pylab import *

def resetSim(context, simdt, plotdt):
    context.setClock(0, simdt, 0)
    context.setClock(1, simdt, 0) #### The hsolve and ee methods use clock 1
    context.setClock(2, simdt, 0) #### hsolve uses clock 2 for mg_block, nmdachan and others.
    context.setClock(PLOTCLOCK, plotdt, 0) # PLOTCLOCK is in mooseConstants.py
    context.reset()

def setupTable(name, obj, qtyname):
    # Setup the tables to pull data
    vmTable = moose.Table(name, moose.Neutral(obj.path+"/data"))
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
    synapse = moose.SynChan(synapseid)
    synapse.Gbar = synapse.Gbar*gbar_factor
    if synapse.getField('mgblock')=='True': # If NMDA synapse based on mgblock, connect to mgblock
        mgblock = moose.Mg_block(synapse.path+'/mgblock')
        compartment.connect("channel", mgblock, "channel")
    else:
        compartment.connect("channel", synapse, "channel")
    return synapse

def connect_CaConc(compartment_list):
    """ Connect the Ca pools and channels within each of the compartments in compartment_list
     Ca channels should have an extra field called 'ion' defined and set in MOOSE.
     Ca dependent channels like KCa should have an extra field called 'ionDependency' defined and set in MOOSE.
     Should call this after instantiating cell so that all channels and pools have been created. """
    context = moose.PyMooseBase.getContext()
    for compartment in compartment_list:
        caconc = None
        for child in compartment.getChildren(compartment.id):
            neutralwrap = moose.Neutral(child)
            if neutralwrap.className == 'CaConc':
                caconc = moose.CaConc(child)
                break
        if caconc is not None:
            for child in compartment.getChildren(compartment.id):
                neutralwrap = moose.Neutral(child)
                if neutralwrap.className == 'HHChannel':
                    channel = moose.HHChannel(child)
                    ### If 'ion' field is not present, the Shell returns '0', cribs and prints out a message but it does not throw an exception
                    if channel.getField('ion') == 'Ca':
                        channel.connect('IkSrc',caconc,'current')
                        #print 'Connected ',channel.path
                if neutralwrap.className == 'HHChannel2D':
                    channel = moose.HHChannel2D(child)
                    ### If 'ionDependency' field is not present, the Shell returns '0', cribs and prints out a message but it does not throw an exception
                    if channel.getField('ionDependency') == 'Ca':
                        caconc.connect('concSrc',channel,'concen')
                        #print 'Connected ',channel.path

def printNetTree():
    root = moose.Neutral('/')
    for id in root.getChildren(root.id): # all subelements of 'root'
        if moose.Neutral(id).className == 'Cell':
            cell = moose.Cell(id)
            print "-------------------- CELL : ",cell.name," ---------------------------"
            printCellTree(cell)

def printCellTree(cell):
    """
    Assumes cells have all their compartments one level below,
    also there should be nothing other than compartments on level below.
    Apart from compartment properties and messages,
    it displays the same for subelements of compartments only one level below the compartments.
    Thus NMDA synapses' mgblock-s will be left out.
    """
    for compartmentid in cell.getChildren(cell.id): # compartments
        comp = moose.Compartment(compartmentid)
        print "  |-",comp.path, 'l=',comp.length, 'd=',comp.diameter, 'Rm=',comp.Rm, 'Ra=',comp.Ra, 'Cm=',comp.Cm, 'EM=',comp.Em
        for inmsg in comp.inMessages():
            print "    |---", inmsg
        for outmsg in comp.outMessages():
            print "    |---", outmsg
        printRecursiveTree(compartmentid, level=2) # for channels and synapses and recursively lower levels

def printRecursiveTree(elementid, level):
    spacefill = '  '*level
    element = moose.Neutral(elementid)
    for childid in element.getChildren(elementid): 
        childobj = moose.Neutral(childid)
        classname = childobj.className
        if classname in ['SynChan','KinSynChan']:
            childobj = moose.SynChan(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'Gbar=',childobj.Gbar
        elif classname in ['HHChannel', 'HHChannel2D']:
            childobj = moose.HHChannel(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'Gbar=',childobj.Gbar, 'Ek=',childobj.Ek
        elif classname in ['CaConc']:
            childobj = moose.CaConc(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'thick=',childobj.thick, 'B=',childobj.B
        elif classname in ['Mg_block']:
            childobj = moose.Mg_block(childid)
            print spacefill+"|--", childobj.name, childobj.className, 'CMg',childobj.CMg, 'KMg_A',childobj.KMg_A, 'KMg_B',childobj.KMg_B
        elif classname in ['Table']: # Table gives segfault if printRecursiveTree is called on it
            return # so go no deeper
        for inmsg in childobj.inMessages():
            print spacefill+"  |---", inmsg
        for outmsg in childobj.outMessages():
            print spacefill+"  |---", outmsg
        if len(childobj.getChildren(childid))>0:
            printRecursiveTree(childid, level+1)

def setup_vclamp(compartment, name, delay1, width1, level1, gain=0.5e-5): #### adapted from squid.g in DEMOS (moose/genesis)
    """
    Typically you need to adjust the PID gain
    For perhaps the Davison 4-compartment mitral or the Davison granule:
    0.5e-5 optimal gain - too high 0.5e-4 drives it to oscillate at high frequency,
    too low 0.5e-6 makes it have an initial overshoot (due to Na channels?)
    """
    moose.Neutral('/elec') # If /elec doesn't exists it creates /elec and returns a reference to it. If it does, it just returns its reference.
    pulsegen = moose.PulseGen('/elec/pulsegen'+name)
    vclamp = moose.DiffAmp('/elec/vclamp'+name)
    vclamp.saturation = 999.0
    vclamp.gain = 1.0
    lowpass = moose.RC('/elec/lowpass'+name)
    lowpass.R = 1.0
    lowpass.C = 50e-6 # 50 microseconds tau
    PID = moose.PIDController('/elec/PID'+name)
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

    vclamp_I = moose.Table("/elec/vClampITable"+name)
    vclamp_I.stepMode = TAB_BUF #TAB_BUF: table acts as a buffer.
    vclamp_I.connect("inputRequest", PID, "output")
    vclamp_I.useClock(PLOTCLOCK)
    
    return vclamp_I

def setup_iclamp(compartment, name, delay1, width1, level1):
    moose.Neutral('/elec') # If /elec doesn't exists it creates /elec and returns a reference to it. If it does, it just returns its reference.
    pulsegen = moose.PulseGen('/elec/pulsegen'+name)
    iclamp = moose.DiffAmp('/elec/iclamp'+name)
    iclamp.saturation = 1e6
    iclamp.gain = 1.0
    pulsegen.trigMode = 0 # free run
    pulsegen.baseLevel = 0.0
    pulsegen.firstDelay = delay1
    pulsegen.firstWidth = width1
    pulsegen.firstLevel = level1
    pulsegen.secondDelay = 1e6
    pulsegen.secondLevel = 0.0
    pulsegen.secondWidth = 0.0
    pulsegen.connect('outputSrc',iclamp,'plusDest')
    iclamp.connect('outputSrc',compartment,'injectMsg')
    return pulsegen

def get_matching_children(parent, names): ## non recursive matching of children with given name.
    matchlist = []
    for childID in parent.children():
        child = moose.Neutral(childID)
        for name in names:
            if name in child.name:
                matchlist.append(childID)
    return matchlist

def underscorize(path): # replace / by underscores in a path
    return string.join(string.split(path,'/'),'_')

def blockChannels(cell, channel_list):
    """
    Sets gmax to zero for channels of the cell specified in channel_list
    Substring matches in channel_list are allowed
    e.g. 'K' should block all K channels (ensure that you don't use capital K elsewhere in your channel name!)
    """
    for compartmentid in cell.getChildren(cell.id): # compartments
        comp = moose.Compartment(compartmentid)
        for childid in comp.getChildren(comp.id):
            child = moose.Neutral(childid)
            if child.className in ['HHChannel', 'HHChannel2D']:
                chan = moose.HHChannel(childid)
                for channame in channel_list:
                    if channame in chan.name:
                        chan.Gbar = 0.0
                    
def attach_spikes(filebase, timetable, mpirank):
    ## read the file that contains all the ORN firing times for this glom, odor and avgnum
    filehandle = open(filebase+'.txt','r')
    spiketimelists = filehandle.readlines()
    filehandle.close()

    filenums = string.split(timetable.getField('fileNumbers'),'_')
    ## Merge all the filenums into a temp file, load it and delete it.
    spiketimes = []
    for filenum in filenums: # loop through file numbers
        timesstr = spiketimelists[int(filenum)]
        if timesstr != '\n':
            timestrlist = string.split(timesstr,' ')
            ## convert to float for sorting else '10.0'<'6.0'
            spiketimes.extend([float(timestr) for timestr in timestrlist])
    spiketimes.sort()
    ## ensure that different processes do not write to the same file by using mpirank
    fn = os.getenv('HOME')+'/tempspikes_'+str(mpirank)+'.txt'
    filehandle = open(fn,'w')
    filehandle.write('\n'.join([str(spiketime) for spiketime in spiketimes]))
    filehandle.close()
    timetable.filename = fn
    os.remove(fn)
