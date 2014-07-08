.. Documentation for all MOOSE builtin functions
.. As visible in the Python module
.. Auto-generated on July 08, 2014


MOOSE Builitin Classes and Functions
====================================
    .. module:: moose

   .. py:class:: vec

      An object uniquely identifying a moose element. moose elements are
      array-like objects which can have one or more single-objects within
      them. vec can be traversed like a Python sequence and is item is an
      element identifying single-objects contained in the array element.
      
      Field access to ematrices are vectorized. For example, vec.name returns a
      tuple containing the names of all the single-elements in this
      vec. There are a few special fields that are unique for vec and are not
      vectorized. These are `path`, `value`, `shape` and `className`.
      There are two ways an vec can be initialized, (1) create a new array
      element or (2) create a reference to an existing object.
      
      
      __init__(self, path=path, n=size, g=isGlobal, dtype=className)
      
      Parameters
      ----------
      path : str 
      Path of an existing array element or for creating a new one. This has
      the same format as unix file path: /{element1}/{element2} ... If there
      is no object with the specified path, moose attempts to create a new
      array element. For that to succeed everything until the last `/`
      character must exist or an error is raised
      
      n : positive int
      This is a positive integers specifying the size of the array element
      to be created. Thus n=2 will create an
      vec with 2 elements.
      
      
      g : int
      Specify if this is a global or local element. Global elements are
      shared between nodes in a computing cluster.
      
      dtype: string
      The vector will be of this moose-class.
      
      
      __init__(self, id)
      
      Create a reference to an existing array object.
      
      Parameters
      ----------
      id : vec/int
      vec of an existing array object. The new object will be another
      reference to this object.

         .. py:method:: delete

            vec.delete() -> None
            
            Delete the underlying moose object. This will invalidate all
            references to this object and any attempt to access it will raise a
            ValueError.

         .. py:method:: getPath

            Return the path of this vec object.

         .. py:method:: getShape

            Get the shape of the vec object as a tuple.

         .. py:method:: getValue

            Return integer representation of the id of the element.

         .. py:method:: setField

            setField(fieldname, value_vector) -> None
            
            Set the value of `fieldname` in all elements under this vec.
            
            Parameters
            ----------
            fieldname: str
                field to be set.
            value: sequence of values
                sequence of values corresponding to individual elements under this
                vec.
            
            Notes
            -----
                This is an interface to SetGet::setVec

   .. py:class:: melement

      Individual moose element contained in an array-type object
      (vec). Each element has a unique path, possibly with its index in
      the vec. These are identified by three components: id_ and
      dindex. id_ is the Id of the containing vec, it has a unique
      numerical value (field `value`). `dindex` is the index of the current
      item in the containing vec. `dindex` is 0 for single elements.
      
          __init__(path, dims, dtype) or
          __init__(id, dataIndex, fieldIndex)
          Initialize moose object
      
          Parameters
          ----------
          path : string
              Target element path.
          dims : tuple or int
              dimensions along each axis (can be        an integer for 1D objects). Default: (1,)
          dtype : string
              the MOOSE class name to be created.
          id : vec or integer
              id of an existing element.

         .. py:method:: connect

            connect(srcfield, destobj, destfield, msgtype) -> bool
            Connect another object via a message.
            Parameters
            ----------
            srcfield : str
                    source field on self.
            destobj : element
                    Destination object to connect to.
            destfield : str
                    field to connect to on `destobj`.
            msgtype : str
                    type of the message. Can be `Single`, `OneToAll`, `AllToOne`,
             `OneToOne`, `Reduce`, `Sparse`. Default: `Single`.
            Returns
            -------
            element of the created message.
            
            See also
            --------
            moose.connect

         .. py:method:: getDataIndex

            getDataIndex()
            
            Return the dataIndex of this object.

         .. py:method:: getField

            getField(fieldName)
            
            Get the value of the field.
            
            Parameters
            ----------
            fieldName : string
                    Name of the field.

         .. py:method:: getFieldIndex

            Get the index of this object as a field.

         .. py:method:: getFieldNames

            getFieldNames(fieldType='')
            
            Get the names of fields on this element.
            
            Parameters
            ----------
            fieldType : str
                    Field type to retrieve. Can be `valueFinfo`, `srcFinfo`,
                    `destFinfo`, `lookupFinfo`, etc. If an empty string is specified,
                    names of all avaialable fields are returned.
            
            Returns
            -------
                    out : tuple of strings.
            
            Example
            -------
            List names of all the source fields in PulseGen class:
            ~~~~
            >>> moose.getFieldNames('PulseGen', 'srcFinfo')
            ('childMsg', 'output')
            ~~~~

         .. py:method:: getFieldType

            getFieldType(fieldName')
            
            Get the string representation of the type of this field.
            
            Parameters
            ----------
            fieldName : string
                    Name of the field to be queried.

         .. py:method:: getId

            getId()
            
            Get the vec of this object

         .. py:method:: getLookupField

            getLookupField(fieldName, key)
            
            Lookup entry for `key` in `fieldName`
            
            Parameters
            ----------
            fieldName : string
                    Name of the lookupfield.
            key : appropriate type for key of the lookupfield (as in the dict getFieldDict).
                    Key for the look-up.

         .. py:method:: getNeighbors

            getNeighbors(fieldName)
            
            Get the objects connected to this element by a message on specified
            field.
            
            Parameters
            ----------
            fieldName : str
                    name of the connection field (a destFinfo or srcFinfo)
            
            Returns
            -------
            out: tuple of ematrices.

         .. py:method:: setDestField

            setDestField(arg0, arg1, ...)
            Set a destination field. This is for advanced uses. destFields can
            (and should) be directly called like functions as
            `element.fieldname(arg0, ...)`
            
            Parameters
            ----------
            The number and type of paramateres depend on the destFinfo to be
            set. Use moose.doc('{classname}.{fieldname}') to get builtin
            documentation on the destFinfo `fieldname`

         .. py:method:: setField

            setField(fieldName, value)
            
            Set the value of specified field.
            
            Parameters
            ----------
            fieldName : string
                    Field to be assigned value to.
            value : python datatype compatible with the type of the field
                    The value to be assigned to the field.

         .. py:method:: setLookupField

            setLookupField(field, key, value)
            Set a lookup field entry.
            Parameters
            ----------
            field : string
                    name of the field to be set
            key : key type
                    key in the lookup field for which the value is to be set.
            value : value type
                    value to be set for `key` in the lookkup field.

         .. py:method:: vec

            Return the vec this element belongs to. This is overridden by the attribute of the same name for quick access.

   .. py:class:: LookupField

      ElementField represents fields that are themselves elements. For
      example, synapse in an IntFire neuron. Element fields can be traversed
      like a sequence. Additionally, you can set the number of entries by
      setting the `num` attribute to a desired value.

         .. py:atribute:: dataIndex

            dataIndex of the field element

         .. py:atribute:: name

            

         .. py:atribute:: num

            Number of entries in the field.

         .. py:atribute:: owner

            

         .. py:atribute:: path

            Path of the field element.

         .. py:atribute:: vec

            Id of the field element.

   .. py:class:: DestField

      DestField is a method field, i.e. it can be called like a function.
      Use moose.doc('classname.fieldname') to display builtin
      documentation for `field` in class `classname`.

   .. py:class:: ElementField

      ElementField represents fields that are themselves elements. For
      example, synapse in an IntFire neuron. Element fields can be traversed
      like a sequence. Additionally, you can set the number of entries by
      setting the `num` attribute to a desired value.

         .. py:atribute:: dataIndex

            dataIndex of the field element

         .. py:atribute:: name

            

         .. py:atribute:: num

            Number of entries in the field.

         .. py:atribute:: owner

            

         .. py:atribute:: path

            Path of the field element.

         .. py:atribute:: vec

            Id of the field element.

   .. py:function:: pwe

      Print present working element. Convenience function for GENESIS
      users. If you want to retrieve the element in stead of printing
      the path, use moose.getCwe()

   .. py:function:: le

      List elements under `el` or current element if no argument
      specified.
      
      Parameters
      ----------
      el : str/melement/vec/None
          The element or the path under which to look. If `None`, children
           of current working element are displayed.
      
      Returns
      -------
      None

   .. py:function:: ce

      Set the current working element. 'ce' is an alias of this function

   .. py:function:: showfield

      Show the fields of the element `el`, their data types and
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
      None

   .. py:function:: showmsg

      Print the incoming and outgoing messages of `el`.
      
      Parameters
      ----------
      el : melement/vec/str
          Object whose messages are to be displayed.
      
      Returns
      -------
      None

   .. py:function:: doc

      Display the documentation for class or field in a class.
      
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

   .. py:function:: element

      moose.element(arg) -> moose object
      
      Convert a path or an object to the appropriate builtin moose class
      instance
      
      Parameters
      ----------
      arg : str/vec/moose object
          path of the moose element to be converted or another element (possibly
          available as a superclass instance).
      
      Returns
      -------
      melement
          MOOSE element (object) corresponding to the `arg` converted to write subclass.

   .. py:function:: getFieldNames

      getFieldNames(className, finfoType='valueFinfo') -> tuple
      
      Get a tuple containing the name of all the fields of `finfoType`
      kind.
      
      Parameters
      ----------
      className : string
          Name of the class to look up.
      finfoType : string
          The kind of field (`valueFinfo`, `srcFinfo`, `destFinfo`,
          `lookupFinfo`, `fieldElementFinfo`.).
      
      Returns
      -------
      tuple
          Names of the fields of type `finfoType` in class `className`.

   .. py:function:: copy

      copy(src, dest, name, n, toGlobal, copyExtMsg) -> bool
      
      Make copies of a moose object.
      
      Parameters
      ----------
      src : vec, element or str
          source object.
      dest : vec, element or str
          Destination object to copy into.
      name : str
          Name of the new object. If omitted, name of the original will be used.
      n : int
          Number of copies to make.
      toGlobal : int
          Relevant for parallel environments only. If false, the copies will
          reside on local node, otherwise all nodes get the copies.
      copyExtMsg : int
          If true, messages to/from external objects are also copied.
      
      Returns
      -------
      vec
          newly copied vec

   .. py:function:: move

      Move a vec object to a destination.

   .. py:function:: delete

      delete(obj)->None
      
      Delete the underlying moose object. This does not delete any of the
      Python objects referring to this vec but does invalidate them. Any
      attempt to access them will raise a ValueError.
      
      Parameters
      ----------
      id : vec
          vec of the object to be deleted.
      
      Returns
      -------
      None

   .. py:function:: useClock

      Schedule objects on a specified clock

   .. py:function:: setClock

      Set the dt of a clock.

   .. py:function:: start

      start(time) -> None
      
      Run simulation for `t` time. Advances the simulator clock by `t`
      time.
      
      After setting up a simulation, YOU MUST CALL MOOSE.REINIT() before
      CALLING MOOSE.START() TO EXECUTE THE SIMULATION. Otherwise, the
      simulator behaviour will be undefined. Once moose.reinit() has been
      called, you can call moose.start(t) as many time as you like. This
      will continue the simulation from the last state for `t` time.
      
      Parameters
      ----------
      t : float
          duration of simulation.
      
      Returns
      --------
          None
      
      See also
      --------
      moose.reinit : (Re)initialize simulation

   .. py:function:: reinit

      reinit() -> None
      
      Reinitialize simulation.
      
      This function (re)initializes moose simulation. It must be called
      before you start the simulation (see moose.start). If you want to
      continue simulation after you have called moose.reinit() and
      moose.start(), you must NOT call moose.reinit() again. Calling
      moose.reinit() again will take the system back to initial setting
      (like clear out all data recording tables, set state variables to
      their initial values, etc.

   .. py:function:: stop

      Stop simulation

   .. py:function:: isRunning

      True if the simulation is currently running.

   .. py:function:: exists

      True if there is an object with specified path.

   .. py:function:: writeSBML

      Export biochemical model to an SBML file.

   .. py:function:: readSBML

      Import SBML model to Moose.

   .. py:function:: loadModel

      loadModel(filename, modelpath, solverclass) -> vec
      
      Load model from a file to a specified path.
      
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

   .. py:function:: saveModel

      saveModel(source, filename) -> None
      
      Save model rooted at `source` to file `filename`.
      
      Parameters
      ----------
      source : vec/element/str
          root of the model tree
      
      filename : str
          destination file to save the model in.
      
      Returns
      -------
      None

   .. py:function:: connect

      connect(src, src_field, dest, dest_field, message_type) -> bool
      
      Create a message between `src_field` on `src` object to `dest_field`
      on `dest` object.
      
      Parameters
      ----------
      src : element/vec/string
          the source object (or its path)
      src_field : str
          the source field name. Fields listed under `srcFinfo` and
          `sharedFinfo` qualify for this.
      dest : element/vec/string
          the destination object.
      dest_field : str
          the destination field name. Fields listed under `destFinfo`
          and `sharedFinfo` qualify for this.
      message_type : str (optional)
          Type of the message. Can be `Single`, `OneToOne`, `OneToAll`.
          If not specified, it defaults to `Single`.
      
      Returns
      -------
      melement
          message-manager for the newly created message.
      
      Example
      -------
      Connect the output of a pulse generator to the input of a spike
      generator::
      
      >>> pulsegen = moose.PulseGen('pulsegen')
      >>> spikegen = moose.SpikeGen('spikegen')
      >>> moose.connect(pulsegen, 'output', spikegen, 'Vm')
      1

   .. py:function:: getCwe

      Get the current working element. 'pwe' is an alias of this function.

   .. py:function:: setCwe

      Set the current working element. 'ce' is an alias of this function

   .. py:function:: getFieldDict

      getFieldDict(className, finfoType) -> dict
      
      Get dictionary of field names and types for specified class.
      
      Parameters
      -----------
      className : str
          MOOSE class to find the fields of.
      finfoType : str (optional)
          Finfo type of the fields to find. If empty or not specified, all
          fields will be retrieved.
      
      Returns
      -------
      dict
          field names and their types.
      
      Notes
      -----
          This behaviour is different from `getFieldNames` where only
          `valueFinfo`s are returned when `finfoType` remains unspecified.
      
      Example
      -------
      List all the source fields on class Neutral::
      
      >>> moose.getFieldDict('Neutral', 'srcFinfo')
      {'childMsg': 'int'}

   .. py:function:: getField

      getField(element, field, fieldtype) -- Get specified field of specified type from object vec.

   .. py:function:: seed

      moose.seed(seedvalue) -> None
      
      Reseed MOOSE random number generator.
      
      Parameters
      ----------
      seed : int
          Optional value to use for seeding. If 0, a random seed is
          automatically created using the current system time and other
          information. If not specified, it defaults to 0.
      
      Returns
      -------
      None

   .. py:function:: rand

      moose.rand() -> [0,1)
      
      
      Returns
      -------
      float in [0, 1) real interval generated by MT19937.

   .. py:function:: wildcardFind

      moose.wildcardFind(expression) -> tuple of melements.
      
      Find an object by wildcard.
      
      Parameters
      ----------
      expression : str
          MOOSE allows wildcard expressions of the form::
      
              {PATH}/{WILDCARD}[{CONDITION}]
      
          where {PATH} is valid path in the element tree.
          {WILDCARD} can be `#` or `##`.
      
          `#` causes the search to be restricted to the children of the
          element specified by {PATH}.
      
          `##` makes the search to recursively go through all the descendants
          of the {PATH} element.
          {CONDITION} can be::
              TYPE={CLASSNAME} : an element satisfies this condition if it is of
              class {CLASSNAME}.
              ISA={CLASSNAME} : alias for TYPE={CLASSNAME}
              CLASS={CLASSNAME} : alias for TYPE={CLASSNAME}
              FIELD({FIELDNAME}){OPERATOR}{VALUE} : compare field {FIELDNAME} with
              {VALUE} by {OPERATOR} where {OPERATOR} is a comparison operator (=,
              !=, >, <, >=, <=).
      
          For example, /mymodel/##[FIELD(Vm)>=-65] will return a list of all
          the objects under /mymodel whose Vm field is >= -65.

   .. py:function:: quit

      Finalize MOOSE threads and quit MOOSE. This is made available for debugging purpose only. It will automatically get called when moose module is unloaded. End user should not use this function.
