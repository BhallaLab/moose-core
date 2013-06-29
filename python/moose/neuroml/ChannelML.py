## Description: class ChannelML for loading ChannelML from file or xml element into MOOSE
## Version 1.0 by Aditya Gilra, NCBS, Bangalore, India, 2011 for serial MOOSE
## Version 1.5 by Niraj Dudani, NCBS, Bangalore, India, 2012, ported to parallel MOOSE
## Version 1.6 by Aditya Gilra, NCBS, Bangalore, India, 2012, minor changes for parallel MOOSE
## Version 1.7 by Aditya Gilra, NCBS, Bangalore, India, 2013, further support for NeuroML 1.8.1

"""
NeuroML.py is the preferred interface. Use this only if NeuroML L1,L2,L3 files are misnamed/scattered.
Instantiate ChannelML class, and thence use method:
readChannelMLFromFile(...) to load a standalone ChannelML file (synapse/channel), OR
readChannelML(...) / readSynapseML to load from an xml.etree xml element (could be part of a larger NeuroML file).
"""

from xml.etree import cElementTree as ET
import string
import os, sys
import math

import moose
from moose.neuroml import utils

class ChannelML():

    def __init__(self,nml_params):
        self.cml='http://morphml.org/channelml/schema'
        self.nml_params = nml_params
        self.temperature = nml_params['temperature']

    def readChannelMLFromFile(self,filename,params={}):
        """ specify params as a dict: e.g. temperature that you need to pass to channels """
        tree = ET.parse(filename)
        channelml_element = tree.getroot()
        for channel in channelml_element.findall('.//{'+self.cml+'}channel_type'):
            ## ideally I should read in extra params from within the channel_type element
            ## and put those in also. Global params should override local ones.
            self.readChannelML(channel,params,channelml_element.attrib['units'])
        for synapse in channelml_element.findall('.//{'+self.cml+'}synapse_type'):
            self.readSynapseML(synapse,channelml_element.attrib['units'])
        for ionConc in channelml_element.findall('.//{'+self.cml+'}ion_concentration'):
            self.readIonConcML(ionConc,channelml_element.attrib['units'])

    def readSynapseML(self,synapseElement,units="SI units"):
        if 'Physiological Units' in units: # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e-3 # S from mS       
        elif 'SI Units' in units:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
        else:
            print "wrong units", units,": exiting ..."
            sys.exit(1)
        moose.Neutral('/library') # creates /library in MOOSE tree; elif present, wraps
        if utils.neuroml_debug: print "loading synapse :",synapseElement.attrib['name'],"into /library ."
        moosesynapse = moose.SynChan('/library/'+synapseElement.attrib['name'])
        doub_exp_syn = synapseElement.find('./{'+self.cml+'}doub_exp_syn')
        moosesynapse.Ek = float(doub_exp_syn.attrib['reversal_potential'])*Vfactor
        moosesynapse.Gbar = float(doub_exp_syn.attrib['max_conductance'])*Gfactor
        moosesynapse.tau1 = float(doub_exp_syn.attrib['rise_time'])*Tfactor # seconds
        moosesynapse.tau2 = float(doub_exp_syn.attrib['decay_time'])*Tfactor # seconds
        ### The delay and weight can be set only after connecting a spike event generator.
        ### delay and weight are arrays: multiple event messages can be connected to a single synapse
        moosesynapse_graded = moose.Mstring(moosesynapse.path+'/graded')
        moosesynapse_graded.value = 'False'
        moosesynapse_mgblock = moose.Mstring(moosesynapse.path+'/mgblock')
        moosesynapse_mgblock.value = 'False'
      
    def readChannelML(self,channelElement,params={},units="SI units"):
        ## I first calculate all functions assuming a consistent system of units.
        ## While filling in the A and B tables, I just convert to SI.
        ## Also convert gmax and Erev.
        if 'Physiological Units' in units: # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e1 # S/m^2 from mS/cm^2  
            concfactor = 1e6 # Mol = mol/m^-3 from mol/cm^-3      
        elif 'SI Units' in units:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
            concfactor = 1.0
        else:
            print "wrong units", units,": exiting ..."
            sys.exit(1)
        moose.Neutral('/library') # creates /library in MOOSE tree; elif present, wraps
        channel_name = channelElement.attrib['name']
        if utils.neuroml_debug: print "loading channel :", channel_name,"into /library ."
        IVrelation = channelElement.find('./{'+self.cml+'}current_voltage_relation')
        concdep = IVrelation.find('./{'+self.cml+'}conc_dependence')
        if concdep is None:
            moosechannel = moose.HHChannel('/library/'+channel_name)
        else:
            moosechannel = moose.HHChannel2D('/library/'+channel_name)
        
        if IVrelation.attrib['cond_law']=="ohmic":
            moosechannel.Gbar = float(IVrelation.attrib['default_gmax']) * Gfactor
            moosechannel.Ek = float(IVrelation.attrib['default_erev']) * Vfactor
            moosechannelIon = moose.Mstring(moosechannel.path+'/ion')
            moosechannelIon.value = IVrelation.attrib['ion']
            if concdep is not None:
                moosechannelIonDependency = moose.Mstring(moosechannel.path+'/ionDependency')
                moosechannelIonDependency.value = concdep.attrib['ion']

        nernstnote = IVrelation.find('./{'+utils.meta_ns+'}notes')
        if nernstnote is not None:
            ## the text in nernstnote is "Nernst,Cout=<float>,z=<int>"
            nernst_params = string.split(nernstnote.text,',')
            if nernst_params[0] == 'Nernst':
                nernstMstring = moose.Mstring(moosechannel.path+'/nernst_str')
                nernstMstring.value = str( float(string.split(nernst_params[1],'=')[1]) * concfactor ) + \
                                        ',' + str( int(string.split(nernst_params[2],'=')[1]) )
        
        gates = IVrelation.findall('./{'+self.cml+'}gate')
        if len(gates)>3:
            print "Sorry! Maximum x, y, and z (three) gates are possible in MOOSE/Genesis"
            sys.exit()
        gate_full_name = [ 'gateX', 'gateY', 'gateZ' ] # These are the names that MOOSE uses to create gates.
        ## if impl_prefs tag is present change VMIN, VMAX and NDIVS
        impl_prefs = channelElement.find('./{'+self.cml+'}impl_prefs')
        if impl_prefs is not None:
            table_settings = impl_prefs.find('./{'+self.cml+'}table_settings')
            ## some problem here... disable
            VMIN_here = float(table_settings.attrib['min_v'])
            VMAX_here = float(table_settings.attrib['max_v'])
            NDIVS_here = int(table_settings.attrib['table_divisions'])
            dv_here = (VMAX_here - VMIN_here) / NDIVS_here
        else:
            ## default VMIN, VMAX and dv are in SI
            ## convert them to current calculation units used by channel definition
            ## while loading into tables, convert them back to SI
            VMIN_here = utils.VMIN/Vfactor
            VMAX_here = utils.VMAX/Vfactor
            NDIVS_here = utils.NDIVS
            dv_here = utils.dv/Vfactor
        offset = IVrelation.find('./{'+self.cml+'}offset')
        if offset is None: vNegOffset = 0.0
        else: vNegOffset = float(offset.attrib['value'])
        self.parameters = []
        for parameter in channelElement.findall('.//{'+self.cml+'}parameter'):
            self.parameters.append( (parameter.attrib['name'],float(parameter.attrib['value'])) )
        
        for num,gate in enumerate(gates):
            # if no q10settings tag, the q10factor remains 1.0
            # if present but no gate attribute, then set q10factor
            # if there is a gate attribute, then set it only if gate attrib matches gate name
            self.q10factor = 1.0
            self.gate_name = gate.attrib['name']
            for q10settings in IVrelation.findall('./{'+self.cml+'}q10_settings'):
                ## self.temperature from neuro.utils
                if 'gate' in q10settings.attrib.keys():
                    if q10settings.attrib['gate'] == self.gate_name:
                        self.setQ10(q10settings)
                        break
                else:
                    self.setQ10(q10settings)

            ############### HHChannel2D crashing on setting Xpower!
            #### temperamental! If you print something before, it gives cannot creategate from copied channel, else crashes
            ## Setting power first. This is necessary because it also
            ## initializes the gate's internal data structures as a side
            ## effect. Alternatively, gates can be initialized explicitly
            ## by calling HHChannel.createGate().
            gate_power = float( gate.get( 'instances' ) )
            if num == 0:
                moosechannel.Xpower = gate_power
                if concdep is not None: moosechannel.Xindex = "VOLT_C1_INDEX"
            elif num == 1:
                moosechannel.Ypower = gate_power
                if concdep is not None: moosechannel.Yindex = "VOLT_C1_INDEX"
            elif num == 2:
                moosechannel.Zpower = gate_power
                if concdep is not None: moosechannel.Zindex = "VOLT_C1_INDEX"
            
            ## Getting handle to gate using the gate's path.
            gate_path = moosechannel.path + '/' + gate_full_name[ num ]
            if concdep is None:
                moosegate = moose.HHGate( gate_path )
                ## set SI values inside MOOSE
                moosegate.min = VMIN_here*Vfactor
                moosegate.max = VMAX_here*Vfactor
                moosegate.divs = NDIVS_here
                ## V.IMP to get smooth curves, else even with 3000 divisions
                ## there are sudden transitions.
                moosegate.useInterpolation = True
            else:
                moosegate = moose.HHGate2D( gate_path )
                        
            ##### If alpha and beta functions exist, make them here
            for transition in gate.findall('./{'+self.cml+'}transition'):
                ## make python functions with names of transitions...
                fn_name = transition.attrib['name']
                ## I assume that transitions if present are called alpha and beta
                ## for forward and backward transitions...
                if fn_name in ['alpha','beta']:
                    self.make_cml_function(transition, fn_name, concdep)
                else:
                    print "Unsupported transition ", name
                    sys.exit()
            
            time_course = gate.find('./{'+self.cml+'}time_course')
            ## tau is divided by self.q10factor in make_function()
            ## thus, it gets divided irrespective of <time_course> tag present or not.
            if time_course is not None:
                self.make_cml_function(time_course, 'tau', concdep)
            steady_state = gate.find('./{'+self.cml+'}steady_state')
            if steady_state is not None:
                self.make_cml_function(steady_state, 'inf', concdep)

            if concdep is None: ca_name = ''                        # no Ca dependence
            else: ca_name = ','+concdep.attrib['variable_name']     # Ca dependence
            
            ## Create tau() and inf() if not present, from alpha() and beta()
            for fn_element,fn_name,fn_expr in [(time_course,'tau',"1/(alpha+beta)"),\
                                                (steady_state,'inf',"alpha/(alpha+beta)")]:
                ## put in args for alpha and beta, could be v and Ca dep.
                expr_string = self.replace(fn_expr, 'alpha', 'self.alpha(v'+ca_name+')')
                expr_string = self.replace(expr_string, 'beta', 'self.beta(v'+ca_name+')')                
                ## if time_course/steady_state are not present,
                ## then alpha annd beta transition elements should be present, and fns created.
                if fn_element is None:
                        self.make_function(fn_name,'generic',expr_string=expr_string,concdep=concdep)

            ## non Ca dependent channel
            if concdep is None:
                ## while calculating, use the units used in xml defn,
                ## while filling in table, I convert to SI units.
                v0 = VMIN_here - vNegOffset
                n_entries = NDIVS_here+1
                tableA = [ 0.0 ] * n_entries
                tableB = [ 0.0 ] * n_entries
                for i in range(n_entries):
                    v = v0 + i * dv_here
                    
                    inf = self.inf(v)
                    tau = self.tau(v)                    
                    ## convert to SI before writing to table
                    ## qfactor is already in inf and tau
                    tableA[i] = inf/tau / Tfactor
                    tableB[i] = 1.0/tau / Tfactor
                
                moosegate.tableA = tableA
                moosegate.tableB = tableB
            
            ## Ca dependent channel
            else:
                ## UNITS: while calculating, use the units used in xml defn,
                ##        while filling in table, I convert to SI units.
                ##        Note here Ca units do not enter, but
                ##         units of CaMIN, CaMAX and ca_conc in fn expr should match.
                v = VMIN_here - vNegOffset
                CaMIN = float(concdep.attrib['min_conc'])
                CaMAX = float(concdep.attrib['max_conc'])
                CaNDIVS = 100
                dCa = (CaMAX-CaMIN)/CaNDIVS
                ## CAREFUL!: tableA = [[0.0]*(CaNDIVS+1)]*(NDIVS_here+1) will not work!
                ## * does a shallow copy, same list will get repeated 200 times!
                ## Thus setting tableA[35][1] = 5.0 will set all rows, 1st col to 5.0!!!!
                tableA = [[0.0]*(CaNDIVS+1) for i in range(NDIVS_here+1)]
                tableB = [[0.0]*(CaNDIVS+1) for i in range(NDIVS_here+1)]
                for i in range(NDIVS_here+1):
                    Ca = CaMIN
                    for j in range(CaNDIVS+1):
                        inf = self.inf(v,Ca)
                        tau = self.tau(v,Ca)
                        ## convert to SI (Tfactor) before writing to table
                        ## qfactor is already in inf and tau
                        tableA[i][j] = inf/tau / Tfactor
                        tableB[i][j] = 1.0/tau / Tfactor
                        Ca += dCa
                    v += dv_here

                ## Presently HHGate2D doesn't allow the setting of tables as 2D vectors directly
                #moosegate.tableA = tableA
                #moosegate.tableB = tableB

                ## Instead, I wrap the interpol2D objects inside HHGate2D and set the tables
                moosegate_tableA = moose.Interpol2D(moosegate.path+'/tableA')
                ## set SI values inside MOOSE
                moosegate_tableA.xmin = VMIN_here*Vfactor
                moosegate_tableA.xmax = VMAX_here*Vfactor
                moosegate_tableA.xdivs = NDIVS_here
                #moosegate_tableA.dx = dv_here*Vfactor
                moosegate_tableA.ymin = CaMIN*concfactor
                moosegate_tableA.ymax = CaMAX*concfactor
                moosegate_tableA.ydivs = CaNDIVS
                #moosegate_tableA.dy = dCa*concfactor
                moosegate_tableA.tableVector2D = tableA

                moosegate_tableB = moose.Interpol2D(moosegate.path+'/tableB')
                ## set SI values inside MOOSE
                moosegate_tableB.xmin = VMIN_here*Vfactor
                moosegate_tableB.xmax = VMAX_here*Vfactor
                moosegate_tableB.xdivs = NDIVS_here
                #moosegate_tableB.dx = dv_here*Vfactor
                moosegate_tableB.ymin = CaMIN*concfactor
                moosegate_tableB.ymax = CaMAX*concfactor
                moosegate_tableB.ydivs = CaNDIVS
                #moosegate_tableB.dy = dCa*concfactor
                moosegate_tableB.tableVector2D = tableB

    def setQ10(self,q10settings):
        if 'q10_factor' in q10settings.attrib.keys():
            self.q10factor = float(q10settings.attrib['q10_factor'])\
                **((self.temperature-float(q10settings.attrib['experimental_temp']))/10.0)
        elif 'fixed_q10' in q10settings.attrib.keys():
            self.q10factor = float(q10settings.attrib['fixed_q10'])


    def readIonConcML(self, ionConcElement, units="SI units"):
        if units == 'Physiological Units': # see pg 219 (sec 13.2) of Book of Genesis
            Vfactor = 1e-3 # V from mV
            Tfactor = 1e-3 # s from ms
            Gfactor = 1e1 # S/m^2 from mS/cm^2
            concfactor = 1e6 # mol/m^3 from mol/cm^3
            Lfactor = 1e-2 # m from cm
            Ifactor = 1e-6 # A from microA
        else:
            Vfactor = 1.0
            Tfactor = 1.0
            Gfactor = 1.0
            concfactor = 1.0
            Lfactor = 1.0
            Ifactor = 1.0
        moose.Neutral('/library') # creates /library in MOOSE tree; elif present, wraps
        ionSpecies = ionConcElement.find('./{'+self.cml+'}ion_species')
        if ionSpecies is not None:
            if not 'ca' in ionSpecies.attrib['name']:
                print "Sorry, I cannot handle non-Ca-ion pools. Exiting ..."
                sys.exit(1)
        capoolName = ionConcElement.attrib['name']
        print "loading Ca pool :",capoolName,"into /library ."
        caPool = moose.CaConc('/library/'+capoolName)
        poolModel = ionConcElement.find('./{'+self.cml+'}decaying_pool_model')
        caPool.CaBasal = float(poolModel.attrib['resting_conc']) * concfactor
        caPool.Ca_base = float(poolModel.attrib['resting_conc']) * concfactor
        if 'decay_constant' in poolModel.attrib:
            caPool.tau = float(poolModel.attrib['decay_constant']) * Tfactor
        elif 'inv_decay_constant' in poolModel.attrib:
            caPool.tau = 1.0/float(poolModel.attrib['inv_decay_constant']) * Tfactor
        ## Only one of pool_volume_info or fixed_pool_info should be present, but not checking
        volInfo = poolModel.find('./{'+self.cml+'}pool_volume_info')
        if volInfo is not None:
            caPool.thick = float(volInfo.attrib['shell_thickness']) * Lfactor
        fixedPoolInfo = poolModel.find('./{'+self.cml+'}fixed_pool_info')
        if fixedPoolInfo is not None:
            ## Put in phi under the caPool, so that it can 
            ## be used instead of thickness to set B (see section 19.2 in Book of Genesis)
            caPool_phi = moose.Mstring(caPool.path+'/phi')
            caPool_phi.value = str( float(fixedPoolInfo.attrib['phi']) * concfactor/Ifactor/Tfactor )
        ## Put a high ceiling and floor for the Ca conc
        #~ self.context.runG('setfield ' + caPool.path + ' ceiling 1e6')
        #~ self.context.runG('setfield ' + caPool.path + ' floor 0.0')

    def make_cml_function(self, element, fn_name, concdep=None):
        fn_type = element.attrib['expr_form']
        if fn_type in ['exponential','sigmoid','exp_linear']:
            fn = self.make_function( fn_name, fn_type, rate=float(element.attrib['rate']),\
                midpoint=float(element.attrib['midpoint']), scale=float(element.attrib['scale'] ) )
        elif fn_type == 'generic':
            ## OOPS! These expressions should be in SI units, since I converted to SI
            ## Ideally I should not convert to SI and have the user consistently use one or the other
            ## Or convert constants in these expressions to SI, only voltage and ca_conc appear!
            ## NO! SEE ABOVE, for calculating values for tables, I use units specified in xml file,
            ## then I convert to SI while writing to MOOSE internal tables.
            expr_string = element.attrib['expr']
            if concdep is None: ca_name = ''                        # no Ca dependence
            else: ca_name = ','+concdep.attrib['variable_name']     # Ca dependence
            expr_string = self.replace(expr_string, 'alpha', 'self.alpha(v'+ca_name+')')
            expr_string = self.replace(expr_string, 'beta', 'self.beta(v'+ca_name+')')                
            fn = self.make_function( fn_name, fn_type, expr_string=expr_string, concdep=concdep )
        else:
            print "Unsupported function type ", fn_type
            sys.exit()
                
    def make_function(self, fn_name, fn_type, **kwargs):
        """ This dynamically creates a function called fn_name
        If fn_type is exponential, sigmoid or exp_linear,
            **kwargs is a dict having keys rate, midpoint and scale.
        If fin_type is generic, **kwargs is a dict having key expr_string """
        if fn_type == 'exponential':
            def fn(self,v):
                return kwargs['rate']*math.exp((v-kwargs['midpoint'])/kwargs['scale'])
        elif fn_type == 'sigmoid':
            def fn(self,v):
                return kwargs['rate'] / ( 1 + math.exp((v-kwargs['midpoint'])/kwargs['scale']) )
        elif fn_type == 'exp_linear':
            def fn(self,v):
                if v-kwargs['midpoint'] == 0.0: return kwargs['rate']
                else:
                    return kwargs['rate'] * ((v-kwargs['midpoint'])/kwargs['scale']) \
                        / ( 1 - math.exp((kwargs['midpoint']-v)/kwargs['scale']) )
        elif fn_type == 'generic':
            ## python cannot evaluate the ternary operator ?:, so evaluate explicitly
            ## for security purposes eval() is not allowed any __builtins__
            ## but only safe functions in utils.safe_dict and
            ## only the local variables/functions v, self
            allowed_locals = {'self':self}
            allowed_locals.update(utils.safe_dict)
            def fn(self,v,ca=None):
                expr_str = kwargs['expr_string']
                allowed_locals['v'] = v
                allowed_locals['celsius'] = self.temperature
                allowed_locals['temp_adj_'+self.gate_name] = self.q10factor
                for i,parameter in enumerate(self.parameters):
                    allowed_locals[parameter[0]] = self.parameters[i][1]
                if kwargs.has_key('concdep'):
                    concdep = kwargs['concdep']
                    ## ca should appear as neuroML defined 'variable_name' to eval()
                    if concdep is not None:
                        allowed_locals[concdep.attrib['variable_name']] = ca
                if '?' in expr_str:
                    condition, alternatives = expr_str.split('?',1)
                    alternativeTrue, alternativeFalse = alternatives.split(':',1)
                    if eval(condition,{"__builtins__":None},allowed_locals):
                        val = eval(alternativeTrue,{"__builtins__":None},allowed_locals)
                    else:
                        val = eval(alternativeFalse,{"__builtins__":None},allowed_locals)
                else:
                    val = eval(expr_str,{"__builtins__":None},allowed_locals)
                if fn_name == 'tau': return val/self.q10factor
                else: return val

        fn.__name__ = fn_name
        setattr(self.__class__, fn.__name__, fn)

    def replace(self, text, findstr, replacestr):
        return string.join(string.split(text,findstr),replacestr)
