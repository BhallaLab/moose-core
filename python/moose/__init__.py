"""pyMOOSE

Python bindings of MOOSE simulator.

References:
-----------

- `Documentation https://moose.readthedocs.io/en/latest/`
- `Development https://github.com/BhallaLab/moose-core`

"""

# Use these guidelines for docstring:
# https://numpydoc.readthedocs.io/en/latest/format.html

import pydoc
import os
import io
import contextlib

# Bring all C++ functions to global namespace. We can overwrite some of these
# methods later.

import moose._moose as _moose

__moose_classes__ = {}


class melement(_moose.ObjId):
    """Base class for all moose classes.
    """

    __type__ = "UNKNOWN"
    __doc__ = ""

    def __init__(self, x, ndata=1, **kwargs):
        obj = _moose.create(self.__type__, x, ndata)
        super().__init__(obj)
        for k, v in kwargs.items():
            super().setField(k, v)


def __to_melement(obj):
    global __moose_classes__
    mc = __moose_classes__[obj.type](obj)
    return mc


# Create MOOSE classes from available Cinfos.
for p in _moose.wildcardFind("/##[TYPE=Cinfo]"):
    cls = type(
        p.name, (melement,), {"__type__": p.name, "__doc__": _moose.generateDoc(p.name)}
    )
    setattr(_moose, cls.__name__, cls)
    __moose_classes__[cls.__name__] = cls


# Import all attributes to global namespace. We must do it here after adding
# class types to _moose.
from moose._moose import *


def version():
    """Reutrns moose version string."""
    return _moose.__version__

def version_info():
    return _moose.version_info()

def about():
    """general information about these bindings.
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
         the source object (or its path).
         (the one that provides information)
     srcfield : str
         source field on self.(type of the information)
     destobj : element
         Destination object to connect to.
         (The one that need to get information)
     destfield : str
         field to connect to on `destobj`
     msgtype : str
         type of the message. It ca be one of the following (default Single).
         - `Single`
         - `OneToAll`  
         - `AllToOne`  
         - `OneToOne`  
         - `Reduce` 
         - `Sparse`  
    
     Returns
     -------
     msgmanager: melement
         message-manager for the newly created message.

     Examples
     --------
     Connect the output of a pulse generator to the input of a spike generator::

     >>> pulsegen = moose.PulseGen('pulsegen')
     >>> spikegen = moose.SpikeGen('spikegen')
     >>> moose.connect(spikegen, 'output', spikegen, 'Vm')
    """
    if isinstance(src, str):
        src = _moose.element(src)
    if isinstance(dest, str):
        dest = _moose.element(dest)
    return src.connect(srcfield, dest, destfield, msgtype)


def delete(arg):
    """Delete the underlying moose object(s). This does not delete any of the
    Python objects referring to this vec but does invalidate them. Any
    attempt to access them will raise a ValueError.

    Parameters
    ----------
    arg : vec/str/melement
        path of the object to be deleted.

    Returns
    -------
    None
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
        field names and their types.

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


def getFieldNames(classname, fieldtype="*"):
    """Get a tuple containing the name of all the fields of `finfoType` kind.

    Parameters
    ----------
    className : string
        Name of the class to look up.
    finfoType : string
        The kind of field
        -  valueFinfo
        -  srcFinfo
        -  destFinfo
        -  lookupFinfo
        -  fieldElementFinfo

    Returns
    -------
    list
        Names of the fields of type `finfoType` in class `className`.
    """
    return _moose.getFieldNames(classname, fieldtype)


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


def loadModelInternal(filename, modelpath, solverclass="gsl"):
    """Load model from a file to a specified path.

    This function should not be used by users. It is meants for developers.
    Please see `moose.loadModel` function.

    Parameters
    ----------
    filename : str
        model description file.
    modelpath : str
        moose path for the top level element of the model to be created.
    solverclass : str, optional
        solver type to be used for simulating the model.

    Returns
    -------
    vec
        loaded model container vec.
    """
    return _moose.loadModelInternal(filename, modelpath, solverclass)


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
    if el is None:
        el = _moose.getCwe()
    elif isinstance(el, str):
        if not _moose.exists(el):
            raise ValueError("no such element")
        el = _moose.element(el)
    elif isinstance(el, _moose.vec):
        el = el[0]
    print("Elements under '%s'" % el)
    for ch in el.children:
        print(" %s" % ch.path)
    return [x.path for x in el.children]


def showfield(el, field="*", showtype=False):
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
                # The following hack is for handling both Python 2 and
                # 3. Directly putting the print command in the if/else
                # clause causes syntax error in both systems.
                result.append(typestr + " ")
            result.append(key.ljust(max_field_len + 4) + "=" + str(value) + "\n")
    else:
        try:
            result.append(field + "=" + el.getField(field))
        except AttributeError:
            pass  # Genesis silently ignores non existent fields
    print("".join(result))
    return "".join(result)


# Predefined field types and their human readable names
__finfotypes = [
    ("valueFinfo", "value field"),
    ("srcFinfo", "source message field"),
    ("destFinfo", "destination message field"),
    ("sharedFinfo", "shared message field"),
    ("lookupFinfo", "lookup field"),
]


def listmsg(el):
    """Return a list containing the incoming and outgoing messages of
    `el`.

    Parameters
    ----------
    el : melement/vec/str
        MOOSE object or path of the object to look into.

    Returns
    -------
    msg : list
        List of Msg objects corresponding to incoming and outgoing
        connections of `el`.

    """
    obj = el
    if isinstance(el, str):
        obj = _moose.element(el)
    ret = []
    for msg in obj.msgIn:
        ret.append(msg)
    for msg in obj.msgOut:
        ret.append(msg)
    return ret


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
    obj = _moose.element(el)
    print("INCOMING:")
    for msg in obj.msgIn:
        print(msg.e2.path, msg.destFieldsOnE2, "<---", msg.e1.path, msg.srcFieldsOnE1)
    print("OUTGOING:")
    for msg in obj.msgOut:
        print(msg.e1.path, msg.srcFieldsOnE1, "--->", msg.e2.path, msg.destFieldsOnE2)


def getFieldDoc(tokens, indent=""):
    """Return the documentation for field specified by `tokens`.

    Parameters
    ----------
    tokens : (className, fieldName) str
        A sequence whose first element is a MOOSE class name and second
        is the field name.

    indent : str
        indentation (default: empty string) prepended to builtin
        documentation string.

    Returns
    -------
    docstring : str
        string of the form
        `{indent}{className}.{fieldName}: {datatype} - {finfoType}\n{Description}\n`

    Raises
    ------
    NameError
        If the specified fieldName is not present in the specified class.
    """
    assert len(tokens) > 1
    classname = tokens[0]
    fieldname = tokens[1]
    while True:
        try:
            classelement = _moose.element("/classes/" + classname)
            for finfo in classelement.children:
                for fieldelement in finfo:
                    baseinfo = ""
                    if classname != tokens[0]:
                        baseinfo = " (inherited from {})".format(classname)
                    if fieldelement.fieldName == fieldname:
                        # The field elements are
                        # /classes/{ParentClass}[0]/{fieldElementType}[N].
                        finfotype = fieldelement.name
                        return u"{indent}{classname}.{fieldname}: type={type}, finfotype={finfotype}{baseinfo}\n\t{docs}\n".format(
                            indent=indent,
                            classname=tokens[0],
                            fieldname=fieldname,
                            type=fieldelement.type,
                            finfotype=finfotype,
                            baseinfo=baseinfo,
                            docs=fieldelement.docs,
                        )
            classname = classelement.baseClass
        except ValueError:
            raise NameError("`%s` has no field called `%s`" % (tokens[0], tokens[1]))


def _appendFinfoDocs(classname, docstring, indent):
    """Append list of finfos in class name to docstring"""
    try:
        classElem = _moose.element("/classes/%s" % (classname))
    except ValueError:
        raise NameError("class '%s' not defined." % (classname))

    for ftype, rname in __finfotypes:
        docstring.write(u"\n*%s*\n" % (rname.capitalize()))
        finfo = _moose.element("%s/%s" % (classElem.path, ftype))
        for field in finfo.vec:
            docstring.write(u"%s%s: %s\n" % (indent, field.fieldName, field.type))


def _getMooseDoc(tokens, inherited=False):
    """Return MOOSE builtin documentation.
    """
    indent = "  "
    docstring = io.StringIO("")
    with contextlib.closing(docstring):
        classElem = _moose.element("/classes/%s" % tokens[0])
        if len(tokens) > 1:
            docstring.write(getFieldDoc(tokens))
            return docstring.getvalue()

        docstring.write(classElem.docs)
        _appendFinfoDocs(tokens[0], docstring, indent)
        return docstring.getvalue()


__pager = None


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
    # There is no way to dynamically access the MOOSE docs using
    # pydoc. (using properties requires copying all the docs strings
    # from MOOSE increasing the loading time by ~3x). Hence we provide a
    # separate function.

    global __pager
    if paged and __pager is None:
        __pager = pydoc.pager
    tokens = []
    if isinstance(arg, str):
        tokens = arg.split(".")
        if tokens[0] in ["moose", "_moose"]:
            tokens = tokens[1:]
    assert tokens
    text = _moose.generateDoc(".".join(tokens))

    if __pager:
        __pager(text)
    else:
        print(text)


# Import from other modules as well.
from moose.server import *
from moose.model_utils import *
