"""pyMOOSE

Python bindings of MOOSE simulator.

References:
-----------

- `Documentation https://moose.readthedocs.io/en/latest/`
- `Development https://github.com/BhallaLab/moose-core`

"""

# Notes
# -----
# Use these guidelines for docstring: https://numpydoc.readthedocs.io/en/latest/format.html

import sys
import pydoc
import os
import typing as T

import moose._moose as _moose
from moose import model_utils

__moose_classes__ : T.Dict[str, T.Any] = {}


class melement(_moose.ObjId):
    """Base class for all moose classes.
    """

    __type__ = "UNKNOWN"
    __doc__ = ""

    def __init__(self, x, ndata=1, **kwargs):
        obj = _moose.__create__(self.__type__, x, ndata)
        if sys.version_info.major > 2:
            super().__init__(obj)
            for k, v in kwargs.items():
                super().setField(k, v)
        else:
            # Support for dead python2.
            super(melement, self).__init__(obj)
            for k, v in kwargs.items():
                super(melement, self).setField(k, v)


def __to_melement(obj):
    global __moose_classes__
    mc = __moose_classes__[obj.type](obj)
    return mc


# Create MOOSE classes from available Cinfos.
for p in _moose.wildcardFind("/##[TYPE=Cinfo]"):
    if sys.version_info.major > 2:
        cls = type(
            p.name,
            (melement,),
            {"__type__": p.name, "__doc__": _moose.__generatedoc__(p.name)},
        )
    else:
        # Python2.
        cls = type(
            str(p.name),
            (melement,),
            {"__type__": p.name, "__doc__": _moose.__generatedoc__(p.name)},
        )
    setattr(_moose, cls.__name__, cls)
    __moose_classes__[cls.__name__] = cls


# Import all attributes to global namespace. We must do it here after adding
# class types to _moose.
from moose._moose import *

def version():
    """Reutrns moose version string."""
    return _moose.__version__

__version__ = version()

def version_info():
    """Return detailed version information.

    >>> moose.version_info()
    {'build_datetime': 'Friday Fri Apr 17 22:13:00 2020',
     'compiler_string': 'GNU,/usr/bin/c++,7.5.0',
     'major': '3',
     'minor': '3',
     'patch': '1'}
    """
    return _moose.version_info()


def about():
    """general information about pyMOOSE.

    Returns
    -------
    A dict

    Example
    -------
    >>> moose.about()
    {'path': '/home/dilawars/Work/GITLAB/moose-core/_build/python/moose',
     'version': '4.0.0.dev20200417',
     'docs': 'https://moose.readthedocs.io/en/latest/',
     'development': 'https://github.com/BhallaLab/moose-core'}
    """
    return dict(
        path=os.path.dirname(__file__),
        version=_moose.__version__,
        docs="https://moose.readthedocs.io/en/latest/",
        development="https://github.com/BhallaLab/moose-core",
    )


def wildcardFind(pattern):
    """Find objects using wildcard pattern

    Parameters
    ----------
    pattern: str
       Wildcard (see note below)

    .. note:: Wildcard

    MOOSE allows wildcard expressions of the form
    {PATH}/{WILDCARD}[{CONDITION}].

    {PATH} is valid path in the element tree, {WILDCARD} can be
    # or ##. # causes the search to be restricted to the children
    of the element specified by {PATH}. ## makes the search to
    recursively go through all the descendants of the {PATH} element.

    {CONDITION} can be:

    - TYPE={CLASSNAME}: an element satisfies this condition if it is of
      class {CLASSNAME}.
    - ISA={CLASSNAME}: alias for TYPE={CLASSNAME}
    - CLASS={CLASSNAME}: alias for TYPE={CLASSNAME}
    - FIELD({FIELDNAME}){OPERATOR}{VALUE} : compare field {FIELDNAME} with
      {VALUE} by {OPERATOR} where {OPERATOR} is a comparison
      operator (=, !=, >, <, >=, <=).

    Returns
    -------
    list
        A list of found MOOSE objects

    Examples
    --------
    Following returns a list of all the objects under /mymodel whose Vm field
    is >= -65.

    >>> moose.wildcardFind("/mymodel/##[FIELD(Vm)>=-65]")
    """
    return [__to_melement(x) for x in _moose.wildcardFind(pattern)]


def connect(src, srcfield, dest, destfield, msgtype="Single"):
    """Create a message between `srcfield` on `src` object to
     `destfield` on `dest` object.

     This function is used mainly, to say, connect two entities, and
     to denote what kind of give-and-take relationship they share.
     It enables the 'destfield' (of the 'destobj') to acquire the
     data, from 'srcfield'(of the 'src').

     Parameters
     ----------
     src : element/vec/string
         the source object (or its path) the one that provides information.
     srcfield : str
         source field on self (type of the information).
     destobj : element
         Destination object to connect to (The one that need to get
         information).
     destfield : str
         field to connect to on `destobj`
     msgtype : str
         type of the message. It ca be one of the following (default Single).
         - Single
         - OneToAll
         - AllToOne
         - OneToOne
         - Reduce
         - Sparse

     Returns
     -------
     msgmanager: melement
         message-manager for the newly created message.

     Note
     -----
     Alternatively, one can also use the following form::

     >>> src.connect(srcfield, dest, destfield, msgtype)


     Examples
     --------
     Connect the output of a pulse generator to the input of a spike generator::

     >>> pulsegen = moose.PulseGen('pulsegen')
     >>> spikegen = moose.SpikeGen('spikegen')
     >>> moose.connect(pulsegen, 'output', spikegen, 'Vm')
     Or,
     >>> pulsegen.connect('output', spikegen, 'Vm')
    """
    src = _moose.element(src)
    dest = _moose.element(dest)
    return src.connect(srcfield, dest, destfield, msgtype)


def delete(arg):
    """Delete the underlying moose object(s). 

    Parameters
    ----------
    arg : vec/str/melement
        path of the object to be deleted.

    Returns
    -------
    None, Raises ValueError if given path/object does not exists.
    """
    _moose.delete(arg)


def element(arg):
    """Convert a path or an object to the appropriate builtin moose class instance

    Parameters
    ----------
    arg : str/vec/moose object
        path of the moose element to be converted or another element (possibly
        available as a superclass instance).

    Returns
    -------
    melement
        MOOSE element (object) corresponding to the `arg` converted to write
        subclass.
    """
    if isinstance(arg, str):
        assert _moose.exists(arg), "Path '%s' does not exists." % arg
    return _moose.element(arg)


def exists(path):
    """Returns True if an object with given path already exists."""
    return _moose.exists(path)


def getCwe():
    """Return current working elemement.

    See also
    --------
    moose.setCwe
    """
    return _moose.getCwe()


def getField(classname, fieldname):
    """Get specified field of specified class."""
    return _moose.getField(classname, fieldname)


def getFieldDict(classname, finfoType=""):
    """Get dictionary of field names and types for specified class.

    Parameters
    ----------
    className : str
        MOOSE class to find the fields of.
    finfoType : str (default '')
        Finfo type of the fields to find. If empty or not specified, allfields
        will be retrieved.

    Returns
    -------
    dict
        field names and their respective types as key-value pair. 

    Notes
    -----
    This behaviour is different from `getFieldNames` where only `valueFinfo`s
    are returned when `finfoType` remains unspecified.

    Examples
    --------
    List all the source fields on class Neutral

    >>> moose.getFieldDict('Neutral', 'srcFinfo')
       {'childMsg': 'int'}
    """
    return _moose.getFieldDict(classname, finfoType)


def getFieldNames(elem, fieldtype="*"):
    """Get a tuple containing name of fields of a given fieldtype. If
    fieldtype is set to '*', all fields are returned.

    Parameters
    ----------
    elem : string, obj
        Name of the class or a moose.element to look up.
    fieldtype : string
        The kind of field. Possible values are:
        -  'valueFinfo' or 'value'
        -  'srcFinfo' or 'src'
        -  'destFinfo' or 'dest'
        -  'lookupFinfo' or 'lookup'
        -  'fieldElementFinfo' or 'fieldElement'

    Returns
    -------
    list
        Names of the fields of type `finfoType` in class `className`.
    """
    clsname = elem if isinstance(elem, str) else elem.className
    return _moose.getFieldNames(clsname, fieldtype)


def isRunning():
    """True if the simulation is currently running."""
    return _moose.isRunning()


def move(src, dest):
    """Move a moose element `src` to destination"""
    return _moose.move(src, dest)


def reinit():
    """Reinitialize simulation.

    This function (re)initializes moose simulation. It must be called before
    you start the simulation (see moose.start). If you want to continue
    simulation after you have called moose.reinit() and moose.start(), you must
    NOT call moose.reinit() again. Calling moose.reinit() again will take the
    system back to initial setting (like clear out all data recording tables,
    set state variables to their initial values, etc.
    """
    _moose.reinit()


def start(runtime, notify=False):
    """Run simulation for `t` time. Advances the simulator clock by `t` time. If
    'notify = True', a message is written to terminal whenever 10\% of
    simulation time is over. \

    After setting up a simulation, YOU MUST CALL MOOSE.REINIT() before CALLING
    MOOSE.START() TO EXECUTE THE SIMULATION. Otherwise, the simulator behaviour
    will be undefined. Once moose.reinit() has been called, you can call
    `moose.start(t)` as many time as you like. This will continue the
    simulation from the last state for `t` time.

    Parameters
    ----------
    t : float
        duration of simulation.
    notify: bool
        default False. If True, notify user whenever 10\% of simultion is over.

    Returns
    -------
        None

    See also
    --------
    moose.reinit : (Re)initialize simulation
    """
    _moose.start(runtime, notify)


def stop():
    """Stop simulation"""
    _moose.stop()


def setCwe(arg):
    """Set the current working element.

    Parameters
    ----------
    arg: str, melement, vec
        moose element or path to be set as cwe.

    See also
    --------
    getCwe
    """
    _moose.setCwe(arg)

def ce(arg):
    """Alias for setCwe"""
    _moose.setCwe(arg)

def useClock(tick, path, fn):
    """Schedule `fn` function of every object that matches `path` on tick no
    `tick`. Usually you don't have to use it.

    (FIXME: Needs update) The sequence of clockticks with the same dt is
    according to their number.  This is utilized for controlling the order of
    updates in various objects where it matters.  The following convention
    should be observed when assigning clockticks to various components of a
    model: Clock ticks 0-3 are for electrical (biophysical) components, 4 and 5
    are for chemical kinetics, 6 and 7 are for lookup tables and stimulus, 8
    and 9 are for recording tables.

    Parameters
    ----------
    tick : int
        tick number on which the targets should be scheduled.
    path : str
        path of the target element(s). This can be a wildcard also.
    fn : str
        name of the function to be called on each tick. Commonly `process`.

    Examples
    --------
    In multi-compartmental neuron model a compartment's membrane potential (Vm)
    is dependent on its neighbours' membrane potential. Thus it must get the
    neighbour's present Vm before computing its own Vm in next time step.  This
    ordering is achieved by scheduling the `init` function, which communicates
    membrane potential, on tick 0 and `process` function on tick 1.

    >>> moose.useClock(0, '/model/compartment_1', 'init')
    >>> moose.useClock(1, '/model/compartment_1', 'process'));
    """
    _moose.useClock(tick, path, fn)


def setClock(clockid, dt):
    """set the ticking interval of `tick` to `dt`.

    A tick with interval `dt` will call the functions scheduled on that tick
    every `dt` timestep.

    Parameters
    ----------
    tick : int
        tick number
    dt : double
        ticking interval

    """
    _moose.setClock(clockid, dt)


def loadModel(filename, modelpath, solverclass="gsl"):
    """loadModel: Load model (genesis/cspace) from a file to a specified path.

    Parameters
    ----------
    filename: str
        model description file.
    modelpath: str
        moose path for the top level element of the model to be created.
    method: str
        solver type to be used for simulating the model.
        TODO: Link to detailed description of solvers?

    Returns
    -------
    melement
        moose.element if succcessful else None.

    See also
    --------
    moose.readNML2
    moose.writeNML2 (NotImplemented)
    moose.readSBML
    moose.writeSBML
    """
    return model_utils.mooseReadKkitGenesis(filename, modelpath, solverclass)


def copy(src, dest, name="", n=1, toGlobal=False, copyExtMsg=False):
    """Make copies of a moose object.

    Parameters
    ----------
    src : vec, element or str
        source object.
    dest : vec, element or str
        Destination object to copy into.
    name : str
        Name of the new object. If omitted, name of the original will be used.
    n : int
        Number of copies to make (default=1).
    toGlobal : bool
        Relevant for parallel environments only. If false, the copies will
        reside on local node, otherwise all nodes get the copies.
    copyExtMsg : bool
        If true, messages to/from external objects are also copied.

    Returns
    -------
    vec
        newly copied vec
    """
    if isinstance(src, str):
        src = _moose.element(src)
    if isinstance(dest, str):
        dest = _moose.element(dest)
    if not name:
        name = src.name
    return _moose.copy(src.id, dest, name, n, toGlobal, copyExtMsg)


def rand(a=0.0, b=1.0):
    """Generate random number from the interval [0.0, 1.0)

    Returns
    -------
    float in [0, 1) real interval generated by MT19937.

    See also
    --------
    moose.seed() : reseed the random number generator.

    Notes
    -----
    MOOSE does not automatically seed the random number generator. You
    must explicitly call moose.seed() to create a new sequence of random
    numbers each time.
    """
    return _moose.rand(a, b)


def seed(seed=0):
    """Reseed MOOSE random number generator.

    Parameters
    ----------
    seed : int
        Value to use for seeding.
        default: random number generated using system random device

    Notes
    -----
    All RNGs in moose except rand functions in moose.Function expression use
    this seed.

    By default (when this function is not called) seed is initializecd to some
    random value using system random device (if available).

    Returns
    -------
    None

    See also
    --------
    moose.rand() : get a pseudorandom number in the [0,1) interval.
    """
    _moose.seed(seed)


def pwe():
    """Print present working element's path.

    Convenience function for GENESIS users. If you want to retrieve the element
    in stead of printing the path, use moose.getCwe().

    Returns
    ------
    melement
        current MOOSE element

    Example
    -------
    >>> pwe()
    '/'
    """
    pwe_ = _moose.getCwe()
    print(pwe_.path)
    return pwe_


def le(el=None):
    """List elements under `el` or current element if no argument
    specified.

    Parameters
    ----------
    el : str/melement/vec/None

        The element or the path under which to look. If `None`, children of
        current working element are displayed.

    Returns
    -------
    List[str]
        path of all children
    """
    el = _moose.getCwe() if el is None else el
    if isinstance(el, str):
        el = _moose.element(el)
    elif isinstance(el, _moose.vec):
        el = el[0]
    return _moose.le(el)


def showfields(el, field="*", showtype=False):
    """Show the fields of the element `el`, their data types and
    values in human readable format. Convenience function for GENESIS
    users.

    Parameters
    ----------
    el : melement/str
        Element or path of an existing element.

    field : str
        Field to be displayed. If '*' (default), all fields are displayed.

    showtype : bool
        If True show the data type of each field. False by default.

    Returns
    -------
    string

    """
    if isinstance(el, str):
        if not _moose.exists(el):
            raise ValueError("no such element: %s" % el)
        el = _moose.element(el)
    result = []
    if field == "*":
        value_field_dict = _moose.getFieldDict(el.className, "valueFinfo")
        max_type_len = max(len(dtype) for dtype in value_field_dict.values())
        max_field_len = max(len(dtype) for dtype in value_field_dict.keys())
        result.append("\n[" + el.path + "]\n")
        for key, dtype in sorted(value_field_dict.items()):
            if (
                dtype == "bad"
                or key == "this"
                or key == "dummy"
                or key == "me"
                or dtype.startswith("vector")
                or "ObjId" in dtype
            ):
                continue
            value = el.getField(key)
            if showtype:
                typestr = dtype.ljust(max_type_len + 4)
                ## The following hack is for handling both Python 2 and
                ## 3. Directly putting the print command in the if/else
                ## clause causes syntax error in both systems.
                result.append(typestr + " ")
            result.append(key.ljust(max_field_len + 4) + "=" + str(value) + "\n")
    else:
        try:
            result.append(field + "=" + el.getField(field))
        except AttributeError:
            pass  # Genesis silently ignores non existent fields
    print("".join(result))
    return "".join(result)


def showfield(el, field="*", showtype=False):
    """Alias for showfields."""
    return showfields(el, field, showtype)


def listmsg(arg):
    """Return a list containing the incoming and outgoing messages of
    `el`.

    Parameters
    ----------
    arg : melement/vec/str
        MOOSE object or path of the object to look into.

    Returns
    -------
    msg : list
        List of Msg objects corresponding to incoming and outgoing connections
        of `arg`.

    """
    obj = _moose.element(arg)
    assert obj
    return _moose.listmsg(obj)


def showmsg(el):
    """Print the incoming and outgoing messages of `el`.

    Parameters
    ----------
    el : melement/vec/str
        Object whose messages are to be displayed.

    Returns
    -------
    None

    """
    print(_moose.showmsg(_moose.element(el)))


def doc(arg, paged=True):
    """Display the documentation for class or field in a class.

    Parameters
    ----------
    arg : str/class/melement/vec
        A string specifying a moose class name and a field name
        separated by a dot. e.g., 'Neutral.name'. Prepending `moose.`
        is allowed. Thus moose.doc('moose.Neutral.name') is equivalent
        to the above.
        It can also be string specifying just a moose class name or a
        moose class or a moose object (instance of melement or vec
        or there subclasses). In that case, the builtin documentation
        for the corresponding moose class is displayed.

    paged: bool
        Whether to display the docs via builtin pager or print and
        exit. If not specified, it defaults to False and
        moose.doc(xyz) will print help on xyz and return control to
        command line.

    Returns
    -------
    None

    Raises
    ------
    NameError
        If class or field does not exist.

    """
    text = _moose.__generatedoc__(arg)
    if pydoc.page:
        pydoc.pager(text) 
    else:
        print(text)


# SBML related functions.
def readSBML(filepath, loadpath, solver="ee", validate=True):
    """Load SBML model.

    Parameters
    ----------
    filepath: str
        filepath to be loaded.
    loadpath : str
        Root path for this model e.g. /model/mymodel
    solver : str
        Solver to use (default 'ee').
        Available options are "ee", "gsl", "stochastic", "gillespie"
            "rk", "deterministic"
            For full list see ??
    validate: bool
        When True, run the schema validation.
    """
    return model_utils.mooseReadSBML(filepath, loadpath, solver, validate)


def writeSBML(modelpath, filepath, sceneitems={}):
    """Writes loaded model under modelpath to a file in SBML format.

    Parameters
    ----------
    modelpath : str
        model path in moose e.g /model/mymodel
    filepath : str
        Path of output file.
    sceneitems : dict
        UserWarning: user need not worry about this layout position is saved in
        Annotation field of all the moose Object (pool,Reaction,enzyme).
        If this function is called from
        * GUI - the layout position of moose object is passed
        * command line - NA
        * if genesis/kkit model is loaded then layout position is taken from the file
        * otherwise auto-coordinates is used for layout position.
    """
    return model_utils.mooseWriteSBML(modelpath, filepath, sceneitems)


def writeKkit(modelpath, filepath, sceneitems={}):
    """Writes  loded model under modelpath to a file in Kkit format.

    Parameters
    ----------
    modelpath : str
        Model path in moose.
    filepath : str
        Path of output file.
    """
    return model_utils.mooseWriteKkit(modelpath, filepath, sceneitems)


def readNML2(modelpath, verbose=False):
    """Load neuroml2 model.

    Parameters
    ----------
    modelpath: str
        Path of nml2 file.

    verbose: True
        (defalt False)
        If True, enable verbose logging.

    Raises
    ------
    FileNotFoundError: If modelpath is not found or not readable.
    """
    return model_utils.mooseReadNML2(modelpath, verbose)


def writeNML2(outfile):
    """Write model to NML2. (Not implemented)
    """
    raise NotImplementedError("Writing to NML2 is not supported yet")


def addChemSolver(modelpath, solver):
    """Add solver on chemical compartment and its children for calculation.
    (For developers)

    Parameters
    ----------
    modelpath : str
        Model path that is loaded into moose.
    solver : str
        Exponential Euler "ee" is default. Other options are Gillespie ("gssa"),
        Runge Kutta ("gsl"/"rk"/"rungekutta").

    TODO
    ----
    Documentation

    See also
    --------
    deleteChemSolver
    """
    return model_utils.mooseAddChemSolver(modelpath, solver)


def deleteChemSolver(modelpath):
    """Deletes solver on all the compartment and its children

    Notes
    -----
    This is neccesary while created a new moose object on a pre-existing modelpath,
    this should be followed by mooseAddChemSolver for add solvers on to compartment
    to simulate else default is Exponential Euler (ee)

    See also
    --------
    addChemSolver
    """
    return model_utils.mooseDeleteChemSolver(modelpath)


def mergeChemModel(modelpath, dest):
    """Merges two models.

    Merge chemical model in a file `modelpath` with existing MOOSE model at
    path `dest`.

    Parameters
    ----------
    modelpath : str
        Filepath containing a chemical model.
    dest : path
        Existing MOOSE path.

    TODO
    ----
        No example file which shows its use. Deprecated?
    """
    return model_utils.mooseMergeChemModel(modelpath, dest)
