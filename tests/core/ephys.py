# ephys.py --- 
# 
# Filename: ephys.py
# Description: 
# Author: Subhasis Ray
# Created: Wed Feb  5 16:00:48 2025 (+0530)
# 

# Code:
"""Utility functions for ephys tests"""
import moose


def create_voltage_clamp(
    compartment,
    modelpath='./elec',
    datapath=None,
    vclampname='vclamp',
    cmdname='command',
):
    """Creates a voltage clamp object under `modelpath` and
    a table under `datapath` to record the command voltage.

    Parameters
    ----------
    compartment: moose.Compartment
        Compartment to be voltage clamped
    modelpath: str (default: './elec')
        Path to container for the electrical circuit
    datapath: moose.melement (default: None)
        Container for data recorders, if `None` the command voltage is not
        recorded, and the returned `command_tab` is also `None`.
    vclampname: str (default: 'vclamp'
        Name of the voltage clamp object.
    cmdname: str (default: 'command')
        Name of the command pulse generator object.

    Returns
    -------
    (vclamp, command, commandtab): `vclamp` is the VoltageClamp object,
        `command` the `moose.PulseGen` that sets the command value to the
        voltage clamp, `command_tab` a `moose.Table` that records the
        command value during simulation. If `datapath` is `None`, then this
        will be `None`.

    """
    # create the elec container if it does not exist
    _ = moose.Neutral(modelpath)
    vclamp_path = f'{modelpath}/{vclampname}'
    existent = False
    if moose.exists(vclamp_path):
        # avoid duplicate connect
        existent = True
        print(
            f'{modelpath}: Object already exists. '
            'Returning references to existing components.'
        )
    _ = moose.Neutral(modelpath)
    vclamp = moose.VClamp(vclamp_path)
    command = moose.PulseGen(f'{modelpath}/{cmdname}')
    # Also setup a table to record the command voltage of the VClamp directly
    if datapath is not None:
        commandtab = moose.Table(f'{datapath}/command')
    else:
        commandtab = None
    if not existent:
        # The voltage clamp's output is `currentOut` which will be
        # injected into the compartment
        moose.connect(vclamp, 'currentOut', compartment, 'injectMsg')
        # The voltage clamp object senses the voltage of the compartment
        moose.connect(compartment, 'VmOut', vclamp, 'sensedIn')
        # Connect the output of the command pulse to the command input of
        # the voltage clamp circuit
        moose.connect(command, 'output', vclamp, 'commandIn')
        if commandtab is not None:
            moose.connect(commandtab, 'requestOut', command, 'getOutputValue')
        # ====================================================
        #     set the parameters for voltage clamp circuit
        # ----------------------------------------------------
        # compartment.dt is the integration time step for the
        # compartment. `tau` is the time constant of the RC filter in
        # the circuit. 5 times the integration timestep value is a
        # good starting point for tau
        vclamp.tau = 5 * compartment.dt
        # `gain` is the proportional gain of the PID
        # controller. `Cm/dt` is a good value
        vclamp.gain = compartment.Cm / compartment.dt
        # `ti` is the integral time of the PID controller, `dt` is a good value
        vclamp.ti = compartment.dt
        # `td` is the derivative time of the PID controller. We can
        # keep it 0, the default value
    return vclamp, command, commandtab


def setup_step_command(command, base, delay, level):
    """Set up an existing pulse generator `command` to output `base`
    as initial value and `level` after `delay` time

    This provides a pulse that is a single step function. If you want
    repeated pulses directly modify the vectors of delay, width, and
    level.

    """
    command.baseLevel = base
    command.firstDelay = delay
    command.secondDelay = 1e9  # Never stop
    command.firstWidth = 1e9  # Never stop
    command.firstLevel = level



# 
# ephys.py ends here
