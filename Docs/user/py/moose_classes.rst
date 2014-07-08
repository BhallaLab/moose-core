.. Documentation for all MOOSE classes and functions
.. As visible in the Python module
.. Auto-generated on July 08, 2014


MOOSE Classes
==================
.. py:class:: Adaptor

   Averages and rescales values to couple different kinds of simulation

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from the scheduler. 

   .. py:method:: setInputOffset

       (*destination message field*) Assigns field value.

   .. py:method:: getInputOffset

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setOutputOffset

       (*destination message field*) Assigns field value.

   .. py:method:: getOutputOffset

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setScale

       (*destination message field*) Assigns field value.

   .. py:method:: getScale

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: input

       (*destination message field*) Input message to the adaptor. If multiple inputs are received, the system averages the inputs.

   .. py:method:: process

       (*destination message field*) Handles 'process' call

   .. py:method:: reinit

       (*destination message field*) Handles 'reinit' call

   .. py:attribute:: output

      double (*source message field*) Sends the output value every timestep.

   .. py:attribute:: requestInput

      void (*source message field*) Sends out the request. Issued from the process call.

   .. py:attribute:: requestField

      Pd (*source message field*) Sends out a request to a generic double field. Issued from the process call.Works for any number of targets.

   .. py:attribute:: inputOffset

      double (*value field*) Offset to apply to input message, before scaling

   .. py:attribute:: outputOffset

      double (*value field*) Offset to apply at output, after scaling

   .. py:attribute:: scale

      double (*value field*) Scaling factor to apply to input

   .. py:attribute:: outputValue

      double (*value field*) This is the linearly transformed output.

.. py:class:: Annotator

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNotes

       (*destination message field*) Assigns field value.

   .. py:method:: getNotes

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setColor

       (*destination message field*) Assigns field value.

   .. py:method:: getColor

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTextColor

       (*destination message field*) Assigns field value.

   .. py:method:: getTextColor

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setIcon

       (*destination message field*) Assigns field value.

   .. py:method:: getIcon

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: x

      double (*value field*) x field. Typically display coordinate x

   .. py:attribute:: y

      double (*value field*) y field. Typically display coordinate y

   .. py:attribute:: z

      double (*value field*) z field. Typically display coordinate z

   .. py:attribute:: notes

      string (*value field*) A string to hold some text notes about parent object

   .. py:attribute:: color

      string (*value field*) A string to hold a text string specifying display color.Can be a regular English color name, or an rgb code rrrgggbbb

   .. py:attribute:: textColor

      string (*value field*) A string to hold a text string specifying color for text labelthat might be on the display for this object.Can be a regular English color name, or an rgb code rrrgggbbb

   .. py:attribute:: icon

      string (*value field*) A string to specify icon to use for display

.. py:class:: Arith

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setFunction

       (*destination message field*) Assigns field value.

   .. py:method:: getFunction

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setOutputValue

       (*destination message field*) Assigns field value.

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getArg1Value

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAnyValue

       (*destination message field*) Assigns field value.

   .. py:method:: getAnyValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: arg1

       (*destination message field*) Handles argument 1. This just assigns it

   .. py:method:: arg2

       (*destination message field*) Handles argument 2. This just assigns it

   .. py:method:: arg3

       (*destination message field*) Handles argument 3. This sums in each input, and clears each clock tick.

   .. py:method:: arg1x2

       (*destination message field*) Store the product of the two arguments in output_

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: output

      double (*source message field*) Sends out the computed value

   .. py:attribute:: function

      string (*value field*) Arithmetic function to perform on inputs.

   .. py:attribute:: outputValue

      double (*value field*) Value of output as computed last timestep.

   .. py:attribute:: arg1Value

      double (*value field*) Value of arg1 as computed last timestep.

   .. py:attribute:: anyValue

      unsigned int,double (*lookup field*) Value of any of the internal fields, output, arg1, arg2, arg3,as specified by the index argument from 0 to 3.

.. py:class:: BufPool

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

.. py:class:: CaConc

   CaConc: Calcium concentration pool. Takes current from a channel and keeps track of calcium buildup and depletion by a single exponential process. 

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process message from scheduler

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setCa

       (*destination message field*) Assigns field value.

   .. py:method:: getCa

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaBasal

       (*destination message field*) Assigns field value.

   .. py:method:: getCaBasal

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCa_base

       (*destination message field*) Assigns field value.

   .. py:method:: getCa_base

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau

       (*destination message field*) Assigns field value.

   .. py:method:: getTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setB

       (*destination message field*) Assigns field value.

   .. py:method:: getB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setThick

       (*destination message field*) Assigns field value.

   .. py:method:: getThick

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCeiling

       (*destination message field*) Assigns field value.

   .. py:method:: getCeiling

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFloor

       (*destination message field*) Assigns field value.

   .. py:method:: getFloor

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: current

       (*destination message field*) Calcium Ion current, due to be converted to conc.

   .. py:method:: currentFraction

       (*destination message field*) Fraction of total Ion current, that is carried by Ca2+.

   .. py:method:: increase

       (*destination message field*) Any input current that increases the concentration.

   .. py:method:: decrease

       (*destination message field*) Any input current that decreases the concentration.

   .. py:method:: basal

       (*destination message field*) Synonym for assignment of basal conc.

   .. py:attribute:: concOut

      double (*source message field*) Concentration of Ca in pool

   .. py:attribute:: Ca

      double (*value field*) Calcium concentration.

   .. py:attribute:: CaBasal

      double (*value field*) Basal Calcium concentration.

   .. py:attribute:: Ca_base

      double (*value field*) Basal Calcium concentration, synonym for CaBasal

   .. py:attribute:: tau

      double (*value field*) Settling time for Ca concentration

   .. py:attribute:: B

      double (*value field*) Volume scaling factor

   .. py:attribute:: thick

      double (*value field*) Thickness of Ca shell.

   .. py:attribute:: ceiling

      double (*value field*) Ceiling value for Ca concentration. If Ca > ceiling, Ca = ceiling. If ceiling <= 0.0, there is no upper limit on Ca concentration value.

   .. py:attribute:: floor

      double (*value field*) Floor value for Ca concentration. If Ca < floor, Ca = floor

.. py:class:: ChanBase

   ChanBase: Base class for assorted ion channels.Presents a common interface for all of them. 

   .. py:attribute:: channel

      void (*shared message field*) This is a shared message to couple channel to compartment. The first entry is a MsgSrc to send Gk and Ek to the compartment The second entry is a MsgDest for Vm from the compartment.

   .. py:attribute:: ghk

      void (*shared message field*) Message to Goldman-Hodgkin-Katz object

   .. py:method:: Vm

       (*destination message field*) Handles Vm message coming in from compartment

   .. py:method:: Vm

       (*destination message field*) Handles Vm message coming in from compartment

   .. py:method:: setGbar

       (*destination message field*) Assigns field value.

   .. py:method:: getGbar

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEk

       (*destination message field*) Assigns field value.

   .. py:method:: getEk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGk

       (*destination message field*) Assigns field value.

   .. py:method:: getGk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: channelOut

      double,double (*source message field*) Sends channel variables Gk and Ek to compartment

   .. py:attribute:: permeabilityOut

      double (*source message field*) Conductance term going out to GHK object

   .. py:attribute:: IkOut

      double (*source message field*) Channel current. This message typically goes to concenobjects that keep track of ion concentration.

   .. py:attribute:: Gbar

      double (*value field*) Maximal channel conductance

   .. py:attribute:: Ek

      double (*value field*) Reversal potential of channel

   .. py:attribute:: Gk

      double (*value field*) Channel conductance variable

   .. py:attribute:: Ik

      double (*value field*) Channel current variable

.. py:class:: ChemCompt

   Pure virtual base class for chemical compartments

   .. py:method:: setVolume

       (*destination message field*) Assigns field value.

   .. py:method:: getVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getVoxelVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getOneVoxelVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumDimensions

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getStencilRate

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getStencilIndex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: buildDefaultMesh

       (*destination message field*) Tells ChemCompt derived class to build a default mesh with thespecified volume and number of meshEntries.

   .. py:method:: setVolumeNotRates

       (*destination message field*) Changes volume but does not notify any child objects.Only works if the ChemCompt has just one voxel.This function will invalidate any concentration term inthe model. If you don't know why you would want to do this,then you shouldn't use this function.

   .. py:method:: resetStencil

       (*destination message field*) Resets the diffusion stencil to the core stencil that only includes the within-mesh diffusion. This is needed prior to building up the cross-mesh diffusion through junctions.

   .. py:method:: setNumMesh

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumMesh

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:attribute:: volume

      double (*value field*) Volume of entire chemical domain.Assigning this only works if the chemical compartment hasonly a single voxel. Otherwise ignored.This function goes through all objects below this on thetree, and rescales their molecule #s and rates as per thevolume change. This keeps concentration the same, and alsomaintains rates as expressed in volume units.

   .. py:attribute:: voxelVolume

      vector<double> (*value field*) Vector of volumes of each of the voxels.

   .. py:attribute:: numDimensions

      unsigned int (*value field*) Number of spatial dimensions of this compartment. Usually 3 or 2

   .. py:attribute:: oneVoxelVolume

      unsigned int,double (*lookup field*) Volume of specified voxel.

   .. py:attribute:: stencilRate

      unsigned int,vector<double> (*lookup field*) vector of diffusion rates in the stencil for specified voxel.The identity of the coupled voxels is given by the partner field 'stencilIndex'.Returns an empty vector for non-voxelized compartments.

   .. py:attribute:: stencilIndex

      unsigned int,vector<unsigned int> (*lookup field*) vector of voxels diffusively coupled to the specified voxel.The diffusion rates into the coupled voxels is given by the partner field 'stencilRate'.Returns an empty vector for non-voxelized compartments.

.. py:class:: Cinfo

   Class information object.

   .. py:method:: getDocs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getBaseClass

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: docs

      string (*value field*) Documentation

   .. py:attribute:: baseClass

      string (*value field*) Name of base class

.. py:class:: Clock

   Clock: Clock class. Handles sequencing of operations in simulations.Every object scheduled for operations in MOOSE is connected to oneof the 'Tick' entries on the Clock.The Clock manages ten 'Ticks', each of which has its own dt,which is an integral multiple of the base clock dt_. On every clock step the ticks are examined to see which of themis due for updating. When a tick is updated, the 'process' call of all the objects scheduled on that tick is called.The default scheduling (should not be overridden) has the following assignment of classes to Ticks:0. Biophysics: Init call on Compartments in EE method1. Biophysics: Channels2. Biophysics: Process call on Compartments3. Undefined 4. Kinetics: Pools, or in ksolve mode: Mesh to handle diffusion5. Kinetics: Reacs, enzymes, etc, or in ksolve mode: Stoich/GSL6. Stimulus tables7. More stimulus tables8. Plots9. Postmaster. This must be called last of all and nothing else should use this Tick. The Postmaster is automatically scheduled at set up time. The Tick should be given the longest possible value, typically but not always equal to one of the other ticks, so as to batch the communications. For spiking-only communications, it is usually possible to space the communication tick by as much as 1-2 ms which is the axonal + synaptic delay. 

   .. py:attribute:: clockControl

      void (*shared message field*) Controls all scheduling aspects of Clock, usually from Shell

   .. py:attribute:: proc0

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc1

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc2

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc3

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc4

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc5

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc6

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc7

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc8

      void (*shared message field*) Shared proc/reinit message

   .. py:attribute:: proc9

      void (*shared message field*) Shared proc/reinit message

   .. py:method:: setDt

       (*destination message field*) Assigns field value.

   .. py:method:: getDt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getRunTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getCurrentTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNsteps

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumTicks

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getCurrentStep

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDts

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIsRunning

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTickStep

       (*destination message field*) Assigns field value.

   .. py:method:: getTickStep

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTickDt

       (*destination message field*) Assigns field value.

   .. py:method:: getTickDt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: start

       (*destination message field*) Sets off the simulation for the specified duration

   .. py:method:: step

       (*destination message field*) Sets off the simulation for the specified # of steps

   .. py:method:: stop

       (*destination message field*) Halts the simulation, with option to restart seamlessly

   .. py:method:: reinit

       (*destination message field*) Zeroes out all ticks, starts at t = 0

   .. py:attribute:: finished

      void (*source message field*) Signal for completion of run

   .. py:attribute:: process0

      PK8ProcInfo (*source message field*) Process for Tick 0

   .. py:attribute:: reinit0

      PK8ProcInfo (*source message field*) Reinit for Tick 0

   .. py:attribute:: process1

      PK8ProcInfo (*source message field*) Process for Tick 1

   .. py:attribute:: reinit1

      PK8ProcInfo (*source message field*) Reinit for Tick 1

   .. py:attribute:: process2

      PK8ProcInfo (*source message field*) Process for Tick 2

   .. py:attribute:: reinit2

      PK8ProcInfo (*source message field*) Reinit for Tick 2

   .. py:attribute:: process3

      PK8ProcInfo (*source message field*) Process for Tick 3

   .. py:attribute:: reinit3

      PK8ProcInfo (*source message field*) Reinit for Tick 3

   .. py:attribute:: process4

      PK8ProcInfo (*source message field*) Process for Tick 4

   .. py:attribute:: reinit4

      PK8ProcInfo (*source message field*) Reinit for Tick 4

   .. py:attribute:: process5

      PK8ProcInfo (*source message field*) Process for Tick 5

   .. py:attribute:: reinit5

      PK8ProcInfo (*source message field*) Reinit for Tick 5

   .. py:attribute:: process6

      PK8ProcInfo (*source message field*) Process for Tick 6

   .. py:attribute:: reinit6

      PK8ProcInfo (*source message field*) Reinit for Tick 6

   .. py:attribute:: process7

      PK8ProcInfo (*source message field*) Process for Tick 7

   .. py:attribute:: reinit7

      PK8ProcInfo (*source message field*) Reinit for Tick 7

   .. py:attribute:: process8

      PK8ProcInfo (*source message field*) Process for Tick 8

   .. py:attribute:: reinit8

      PK8ProcInfo (*source message field*) Reinit for Tick 8

   .. py:attribute:: process9

      PK8ProcInfo (*source message field*) Process for Tick 9

   .. py:attribute:: reinit9

      PK8ProcInfo (*source message field*) Reinit for Tick 9

   .. py:attribute:: dt

      double (*value field*) Base timestep for simulation

   .. py:attribute:: runTime

      double (*value field*) Duration to run the simulation

   .. py:attribute:: currentTime

      double (*value field*) Current simulation time

   .. py:attribute:: nsteps

      unsigned int (*value field*) Number of steps to advance the simulation, in units of the smallest timestep on the clock ticks

   .. py:attribute:: numTicks

      unsigned int (*value field*) Number of clock ticks

   .. py:attribute:: currentStep

      unsigned int (*value field*) Current simulation step

   .. py:attribute:: dts

      vector<double> (*value field*) Utility function returning the dt (timestep) of all ticks.

   .. py:attribute:: isRunning

      bool (*value field*) Utility function to report if simulation is in progress.

   .. py:attribute:: tickStep

      unsigned int,unsigned int (*lookup field*) Step size of specified Tick, as integral multiple of dt_ A zero step size means that the Tick is inactive

   .. py:attribute:: tickDt

      unsigned int,double (*lookup field*) Timestep dt of specified Tick. Always integral multiple of dt_. If you assign a non-integer multiple it will round off.  A zero timestep means that the Tick is inactive

.. py:class:: Compartment

   Compartment object, for branching neuron models.

.. py:class:: CompartmentBase

   CompartmentBase object, for branching neuron models.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects. The Process should be called _second_ in each clock tick, after the Init message.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:attribute:: init

      void (*shared message field*) This is a shared message to receive Init messages from the scheduler objects. Its job is to separate the compartmental calculations from the message passing. It doesn't really need to be shared, as it does not use the reinit part, but the scheduler objects expect this form of message for all scheduled output. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a dummy MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:attribute:: channel

      void (*shared message field*) This is a shared message from a compartment to channels. The first entry is a MsgDest for the info coming from the channel. It expects Gk and Ek from the channel as args. The second entry is a MsgSrc sending Vm 

   .. py:attribute:: axial

      void (*shared message field*) This is a shared message between asymmetric compartments. axial messages (this kind) connect up to raxial messages (defined below). The soma should use raxial messages to connect to the axial message of all the immediately adjacent dendritic compartments.This puts the (low) somatic resistance in series with these dendrites. Dendrites should then use raxial messages toconnect on to more distal dendrites. In other words, raxial messages should face outward from the soma. The first entry is a MsgSrc sending Vm to the axialFuncof the target compartment. The second entry is a MsgDest for the info coming from the other compt. It expects Ra and Vm from the other compt as args. Note that the message is named after the source type. 

   .. py:attribute:: raxial

      void (*shared message field*) This is a raxial shared message between asymmetric compartments. The first entry is a MsgDest for the info coming from the other compt. It expects Vm from the other compt as an arg. The second is a MsgSrc sending Ra and Vm to the raxialFunc of the target compartment. 

   .. py:method:: setVm

       (*destination message field*) Assigns field value.

   .. py:method:: getVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCm

       (*destination message field*) Assigns field value.

   .. py:method:: getCm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEm

       (*destination message field*) Assigns field value.

   .. py:method:: getEm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInject

       (*destination message field*) Assigns field value.

   .. py:method:: getInject

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInitVm

       (*destination message field*) Assigns field value.

   .. py:method:: getInitVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRm

       (*destination message field*) Assigns field value.

   .. py:method:: getRm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRa

       (*destination message field*) Assigns field value.

   .. py:method:: getRa

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDiameter

       (*destination message field*) Assigns field value.

   .. py:method:: getDiameter

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLength

       (*destination message field*) Assigns field value.

   .. py:method:: getLength

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX0

       (*destination message field*) Assigns field value.

   .. py:method:: getX0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY0

       (*destination message field*) Assigns field value.

   .. py:method:: getY0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ0

       (*destination message field*) Assigns field value.

   .. py:method:: getZ0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: injectMsg

       (*destination message field*) The injectMsg corresponds to the INJECT message in the GENESIS compartment. Unlike the 'inject' field, any value assigned by handleInject applies only for a single timestep.So it needs to be updated every dt for a steady (or varying)injection current

   .. py:method:: randInject

       (*destination message field*) Sends a random injection current to the compartment. Must beupdated each timestep.Arguments to randInject are probability and current.

   .. py:method:: injectMsg

       (*destination message field*) The injectMsg corresponds to the INJECT message in the GENESIS compartment. Unlike the 'inject' field, any value assigned by handleInject applies only for a single timestep.So it needs to be updated every dt for a steady (or varying)injection current

   .. py:method:: cable

       (*destination message field*) Message for organizing compartments into groups, calledcables. Doesn't do anything.

   .. py:method:: process

       (*destination message field*) Handles 'process' call

   .. py:method:: reinit

       (*destination message field*) Handles 'reinit' call

   .. py:method:: initProc

       (*destination message field*) Handles Process call for the 'init' phase of the CompartmentBase calculations. These occur as a separate Tick cycle from the regular proc cycle, and should be called before the proc msg.

   .. py:method:: initReinit

       (*destination message field*) Handles Reinit call for the 'init' phase of the CompartmentBase calculations.

   .. py:method:: handleChannel

       (*destination message field*) Handles conductance and Reversal potential arguments from Channel

   .. py:method:: handleRaxial

       (*destination message field*) Handles Raxial info: arguments are Ra and Vm.

   .. py:method:: handleAxial

       (*destination message field*) Handles Axial information. Argument is just Vm.

   .. py:attribute:: VmOut

      double (*source message field*) Sends out Vm value of compartment on each timestep

   .. py:attribute:: axialOut

      double (*source message field*) Sends out Vm value of compartment to adjacent compartments,on each timestep

   .. py:attribute:: raxialOut

      double,double (*source message field*) Sends out Raxial information on each timestep, fields are Ra and Vm

   .. py:attribute:: Vm

      double (*value field*) membrane potential

   .. py:attribute:: Cm

      double (*value field*) Membrane capacitance

   .. py:attribute:: Em

      double (*value field*) Resting membrane potential

   .. py:attribute:: Im

      double (*value field*) Current going through membrane

   .. py:attribute:: inject

      double (*value field*) Current injection to deliver into compartment

   .. py:attribute:: initVm

      double (*value field*) Initial value for membrane potential

   .. py:attribute:: Rm

      double (*value field*) Membrane resistance

   .. py:attribute:: Ra

      double (*value field*) Axial resistance of compartment

   .. py:attribute:: diameter

      double (*value field*) Diameter of compartment

   .. py:attribute:: length

      double (*value field*) Length of compartment

   .. py:attribute:: x0

      double (*value field*) X coordinate of start of compartment

   .. py:attribute:: y0

      double (*value field*) Y coordinate of start of compartment

   .. py:attribute:: z0

      double (*value field*) Z coordinate of start of compartment

   .. py:attribute:: x

      double (*value field*) x coordinate of end of compartment

   .. py:attribute:: y

      double (*value field*) y coordinate of end of compartment

   .. py:attribute:: z

      double (*value field*) z coordinate of end of compartment

.. py:class:: CplxEnzBase

   :		Base class for mass-action enzymes in which there is an  explicit pool for the enzyme-substrate complex. It models the reaction: E + S <===> E.S ----> E + P

   .. py:attribute:: enz

      void (*shared message field*) Connects to enzyme pool

   .. py:attribute:: cplx

      void (*shared message field*) Connects to enz-sub complex pool

   .. py:method:: setK1

       (*destination message field*) Assigns field value.

   .. py:method:: getK1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setK2

       (*destination message field*) Assigns field value.

   .. py:method:: getK2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setK3

       (*destination message field*) Assigns field value.

   .. py:method:: getK3

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRatio

       (*destination message field*) Assigns field value.

   .. py:method:: getRatio

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setConcK1

       (*destination message field*) Assigns field value.

   .. py:method:: getConcK1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: enzDest

       (*destination message field*) Handles # of molecules of Enzyme

   .. py:method:: cplxDest

       (*destination message field*) Handles # of molecules of enz-sub complex

   .. py:attribute:: enzOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: cplxOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: k1

      double (*value field*) Forward reaction from enz + sub to complex, in # units.This parameter is subordinate to the Km. This means thatwhen Km is changed, this changes. It also means that whenk2 or k3 (aka kcat) are changed, we assume that Km remainsfixed, and as a result k1 must change. It is only whenk1 is assigned directly that we assume that the user knowswhat they are doing, and we adjust Km accordingly.k1 is also subordinate to the 'ratio' field, since setting the ratio reassigns k2.Should you wish to assign the elementary rates k1, k2, k3,of an enzyme directly, always assign k1 last.

   .. py:attribute:: k2

      double (*value field*) Reverse reaction from complex to enz + sub

   .. py:attribute:: k3

      double (*value field*) Forward rate constant from complex to product + enz

   .. py:attribute:: ratio

      double (*value field*) Ratio of k2/k3

   .. py:attribute:: concK1

      double (*value field*) K1 expressed in concentration (1/millimolar.sec) unitsThis parameter is subordinate to the Km. This means thatwhen Km is changed, this changes. It also means that whenk2 or k3 (aka kcat) are changed, we assume that Km remainsfixed, and as a result concK1 must change. It is only whenconcK1 is assigned directly that we assume that the user knowswhat they are doing, and we adjust Km accordingly.concK1 is also subordinate to the 'ratio' field, sincesetting the ratio reassigns k2.Should you wish to assign the elementary rates concK1, k2, k3,of an enzyme directly, always assign concK1 last.

.. py:class:: CubeMesh

   .. py:method:: setIsToroid

       (*destination message field*) Assigns field value.

   .. py:method:: getIsToroid

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setPreserveNumEntries

       (*destination message field*) Assigns field value.

   .. py:method:: getPreserveNumEntries

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAlwaysDiffuse

       (*destination message field*) Assigns field value.

   .. py:method:: getAlwaysDiffuse

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX0

       (*destination message field*) Assigns field value.

   .. py:method:: getX0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY0

       (*destination message field*) Assigns field value.

   .. py:method:: getY0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ0

       (*destination message field*) Assigns field value.

   .. py:method:: getZ0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX1

       (*destination message field*) Assigns field value.

   .. py:method:: getX1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY1

       (*destination message field*) Assigns field value.

   .. py:method:: getY1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ1

       (*destination message field*) Assigns field value.

   .. py:method:: getZ1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDx

       (*destination message field*) Assigns field value.

   .. py:method:: getDx

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDy

       (*destination message field*) Assigns field value.

   .. py:method:: getDy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDz

       (*destination message field*) Assigns field value.

   .. py:method:: getDz

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNx

       (*destination message field*) Assigns field value.

   .. py:method:: getNx

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNy

       (*destination message field*) Assigns field value.

   .. py:method:: getNy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNz

       (*destination message field*) Assigns field value.

   .. py:method:: getNz

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCoords

       (*destination message field*) Assigns field value.

   .. py:method:: getCoords

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMeshToSpace

       (*destination message field*) Assigns field value.

   .. py:method:: getMeshToSpace

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSpaceToMesh

       (*destination message field*) Assigns field value.

   .. py:method:: getSpaceToMesh

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSurface

       (*destination message field*) Assigns field value.

   .. py:method:: getSurface

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: isToroid

      bool (*value field*) Flag. True when the mesh should be toroidal, that is,when going beyond the right face brings us around to theleft-most mesh entry, and so on. If we have nx, ny, nzentries, this rule means that the coordinate (x, ny, z)will map onto (x, 0, z). Similarly,(-1, y, z) -> (nx-1, y, z)Default is false

   .. py:attribute:: preserveNumEntries

      bool (*value field*) Flag. When it is true, the numbers nx, ny, nz remainunchanged when x0, x1, y0, y1, z0, z1 are altered. Thusdx, dy, dz would change instead. When it is false, thendx, dy, dz remain the same and nx, ny, nz are altered.Default is true

   .. py:attribute:: alwaysDiffuse

      bool (*value field*) Flag. When it is true, the mesh matches up sequential mesh entries for diffusion and chmestry. This is regardless of spatial location, and is guaranteed to set up at least the home reaction systemDefault is false

   .. py:attribute:: x0

      double (*value field*) X coord of one end

   .. py:attribute:: y0

      double (*value field*) Y coord of one end

   .. py:attribute:: z0

      double (*value field*) Z coord of one end

   .. py:attribute:: x1

      double (*value field*) X coord of other end

   .. py:attribute:: y1

      double (*value field*) Y coord of other end

   .. py:attribute:: z1

      double (*value field*) Z coord of other end

   .. py:attribute:: dx

      double (*value field*) X size for mesh

   .. py:attribute:: dy

      double (*value field*) Y size for mesh

   .. py:attribute:: dz

      double (*value field*) Z size for mesh

   .. py:attribute:: nx

      unsigned int (*value field*) Number of subdivisions in mesh in X

   .. py:attribute:: ny

      unsigned int (*value field*) Number of subdivisions in mesh in Y

   .. py:attribute:: nz

      unsigned int (*value field*) Number of subdivisions in mesh in Z

   .. py:attribute:: coords

      vector<double> (*value field*) Set all the coords of the cuboid at once. Order is:x0 y0 z0   x1 y1 z1   dx dy dzWhen this is done, it recalculates the numEntries since dx, dy and dz are given explicitly.As a special hack, you can leave out dx, dy and dz and use a vector of size 6. In this case the operation assumes that nx, ny and nz are to be preserved and dx, dy and dz will be recalculated. 

   .. py:attribute:: meshToSpace

      vector<unsigned int> (*value field*) Array in which each mesh entry stores spatial (cubic) index

   .. py:attribute:: spaceToMesh

      vector<unsigned int> (*value field*) Array in which each space index (obtained by linearizing the xyz coords) specifies which meshIndex is present.In many cases the index will store the EMPTY flag if there isno mesh entry at that spatial location

   .. py:attribute:: surface

      vector<unsigned int> (*value field*) Array specifying surface of arbitrary volume within the CubeMesh. All entries must fall within the cuboid. Each entry of the array is a spatial index obtained by linearizing the ix, iy, iz coordinates within the cuboid. So, each entry == ( iz * ny + iy ) * nx + ixNote that the voxels listed on the surface are WITHIN the volume of the CubeMesh object

.. py:class:: CylMesh

   .. py:method:: setX0

       (*destination message field*) Assigns field value.

   .. py:method:: getX0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY0

       (*destination message field*) Assigns field value.

   .. py:method:: getY0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ0

       (*destination message field*) Assigns field value.

   .. py:method:: getZ0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setR0

       (*destination message field*) Assigns field value.

   .. py:method:: getR0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX1

       (*destination message field*) Assigns field value.

   .. py:method:: getX1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY1

       (*destination message field*) Assigns field value.

   .. py:method:: getY1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ1

       (*destination message field*) Assigns field value.

   .. py:method:: getZ1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setR1

       (*destination message field*) Assigns field value.

   .. py:method:: getR1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDiffLength

       (*destination message field*) Assigns field value.

   .. py:method:: getDiffLength

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCoords

       (*destination message field*) Assigns field value.

   .. py:method:: getCoords

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumDiffCompts

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getTotLength

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: x0

      double (*value field*) x coord of one end

   .. py:attribute:: y0

      double (*value field*) y coord of one end

   .. py:attribute:: z0

      double (*value field*) z coord of one end

   .. py:attribute:: r0

      double (*value field*) Radius of one end

   .. py:attribute:: x1

      double (*value field*) x coord of other end

   .. py:attribute:: y1

      double (*value field*) y coord of other end

   .. py:attribute:: z1

      double (*value field*) z coord of other end

   .. py:attribute:: r1

      double (*value field*) Radius of other end

   .. py:attribute:: diffLength

      double (*value field*) Length constant to use for subdivisionsThe system will attempt to subdivide using compartments oflength diffLength on average. If the cylinder has different enddiameters r0 and r1, it will scale to smaller lengthsfor the smaller diameter end and vice versa.Once the value is set it will recompute diffLength as totLength/numEntries

   .. py:attribute:: coords

      vector<double> (*value field*) All the coords as a single vector: x0 y0 z0  x1 y1 z1  r0 r1 diffLength

   .. py:attribute:: numDiffCompts

      unsigned int (*value field*) Number of diffusive compartments in model

   .. py:attribute:: totLength

      double (*value field*) Total length of cylinder

.. py:class:: DiagonalMsg

   .. py:method:: setStride

       (*destination message field*) Assigns field value.

   .. py:method:: getStride

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: stride

      int (*value field*) The stride is the increment to the src DataId that gives thedest DataId. It can be positive or negative, but bounds checkingtakes place and it does not wrap around.

.. py:class:: DifShell

   DifShell object: Models diffusion of an ion (typically calcium) within an electric compartment. A DifShell is an iso-concentration region with respect to the ion. Adjoining DifShells exchange flux of this ion, and also keep track of changes in concentration due to pumping, buffering and channel currents, by talking to the appropriate objects.

   .. py:attribute:: process_0

      void (*shared message field*) Here we create 2 shared finfos to attach with the Ticks. This is because we want to perform DifShell computations in 2 stages, much as in the Compartment object. In the first stage we send out the concentration value to other DifShells and Buffer elements. We also receive fluxes and currents and sum them up to compute ( dC / dt ). In the second stage we find the new C value using an explicit integration method. This 2-stage procedure eliminates the need to store and send prev_C values, as was common in GENESIS.

   .. py:attribute:: process_1

      void (*shared message field*) Second process call

   .. py:attribute:: buffer

      void (*shared message field*) This is a shared message from a DifShell to a Buffer (FixBuffer or DifBuffer). During stage 0:
 - DifShell sends ion concentration 
- Buffer updates buffer concentration and sends it back immediately using a call-back.
- DifShell updates the time-derivative ( dC / dt ) 
During stage 1: 
- DifShell advances concentration C 
This scheme means that the Buffer does not need to be scheduled, and it does its computations when it receives a cue from the DifShell. May not be the best idea, but it saves us from doing the above computations in 3 stages instead of 2.

   .. py:attribute:: innerDif

      void (*shared message field*) This shared message (and the next) is between DifShells: adjoining shells exchange information to find out the flux between them. Using this message, an inner shell sends to, and receives from its outer shell.

   .. py:attribute:: outerDif

      void (*shared message field*) Using this message, an outer shell sends to, and receives from its inner shell.

   .. py:method:: getC

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCeq

       (*destination message field*) Assigns field value.

   .. py:method:: getCeq

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setD

       (*destination message field*) Assigns field value.

   .. py:method:: getD

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setValence

       (*destination message field*) Assigns field value.

   .. py:method:: getValence

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLeak

       (*destination message field*) Assigns field value.

   .. py:method:: getLeak

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setShapeMode

       (*destination message field*) Assigns field value.

   .. py:method:: getShapeMode

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLength

       (*destination message field*) Assigns field value.

   .. py:method:: getLength

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDiameter

       (*destination message field*) Assigns field value.

   .. py:method:: getDiameter

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setThickness

       (*destination message field*) Assigns field value.

   .. py:method:: getThickness

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVolume

       (*destination message field*) Assigns field value.

   .. py:method:: getVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setOuterArea

       (*destination message field*) Assigns field value.

   .. py:method:: getOuterArea

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInnerArea

       (*destination message field*) Assigns field value.

   .. py:method:: getInnerArea

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Reinit happens only in stage 0

   .. py:method:: process

       (*destination message field*) Handle process call

   .. py:method:: reinit

       (*destination message field*) Reinit happens only in stage 0

   .. py:method:: reaction

       (*destination message field*) Here the DifShell receives reaction rates (forward and backward), and concentrations for the free-buffer and bound-buffer molecules.

   .. py:method:: fluxFromOut

       (*destination message field*) Destination message

   .. py:method:: fluxFromIn

       (*destination message field*) 

   .. py:method:: influx

       (*destination message field*) 

   .. py:method:: outflux

       (*destination message field*) 

   .. py:method:: fInflux

       (*destination message field*) 

   .. py:method:: fOutflux

       (*destination message field*) 

   .. py:method:: storeInflux

       (*destination message field*) 

   .. py:method:: storeOutflux

       (*destination message field*) 

   .. py:method:: tauPump

       (*destination message field*) 

   .. py:method:: eqTauPump

       (*destination message field*) 

   .. py:method:: mmPump

       (*destination message field*) 

   .. py:method:: hillPump

       (*destination message field*) 

   .. py:attribute:: concentrationOut

      double (*source message field*) Sends out concentration

   .. py:attribute:: innerDifSourceOut

      double,double (*source message field*) Sends out source information.

   .. py:attribute:: outerDifSourceOut

      double,double (*source message field*) Sends out source information.

   .. py:attribute:: C

      double (*value field*) Concentration C is computed by the DifShell and is read-only

   .. py:attribute:: Ceq

      double (*value field*) 

   .. py:attribute:: D

      double (*value field*) 

   .. py:attribute:: valence

      double (*value field*) 

   .. py:attribute:: leak

      double (*value field*) 

   .. py:attribute:: shapeMode

      unsigned int (*value field*) 

   .. py:attribute:: length

      double (*value field*) 

   .. py:attribute:: diameter

      double (*value field*) 

   .. py:attribute:: thickness

      double (*value field*) 

   .. py:attribute:: volume

      double (*value field*) 

   .. py:attribute:: outerArea

      double (*value field*) 

   .. py:attribute:: innerArea

      double (*value field*) 

.. py:class:: DiffAmp

   A difference amplifier. Output is the difference between the total plus inputs and the total minus inputs multiplied by gain. Gain can be set statically as a field or can be a destination message and thus dynamically determined by the output of another object. Same as GENESIS diffamp object.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: setGain

       (*destination message field*) Assigns field value.

   .. py:method:: getGain

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSaturation

       (*destination message field*) Assigns field value.

   .. py:method:: getSaturation

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: gainIn

       (*destination message field*) Destination message to control gain dynamically.

   .. py:method:: plusIn

       (*destination message field*) Positive input terminal of the amplifier. All the messages connected here are summed up to get total positive input.

   .. py:method:: minusIn

       (*destination message field*) Negative input terminal of the amplifier. All the messages connected here are summed up to get total positive input.

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: output

      double (*source message field*) Current output level.

   .. py:attribute:: gain

      double (*value field*) Gain of the amplifier. The output of the amplifier is the difference between the totals in plus and minus inputs multiplied by the gain. Defaults to 1

   .. py:attribute:: saturation

      double (*value field*) Saturation is the bound on the output. If output goes beyond the +/-saturation range, it is truncated to the closer of +saturation and -saturation. Defaults to the maximum double precision floating point number representable on the system.

   .. py:attribute:: outputValue

      double (*value field*) Output of the amplifier, i.e. gain * (plus - minus).

.. py:class:: Double

   Variable for storing values.

   .. py:method:: setValue

       (*destination message field*) Assigns field value.

   .. py:method:: getValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: value

      double (*value field*) Variable value

.. py:class:: Dsolve

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setStoich

       (*destination message field*) Assigns field value.

   .. py:method:: getStoich

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setPath

       (*destination message field*) Assigns field value.

   .. py:method:: getPath

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCompartment

       (*destination message field*) Assigns field value.

   .. py:method:: getCompartment

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumAllVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNVec

       (*destination message field*) Assigns field value.

   .. py:method:: getNVec

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumPools

       (*destination message field*) Assigns field value.

   .. py:method:: getNumPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: buildNeuroMeshJunctions

       (*destination message field*) Builds junctions between NeuroMesh, SpineMesh and PsdMesh

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: stoich

      Id (*value field*) Stoichiometry object for handling this reaction system.

   .. py:attribute:: path

      string (*value field*) Path of reaction system. Must include all the pools that are to be handled by the Dsolve, can also include other random objects, which will be ignored.

   .. py:attribute:: compartment

      Id (*value field*) Reac-diff compartment in which this diffusion system is embedded.

   .. py:attribute:: numVoxels

      unsigned int (*value field*) Number of voxels in the core reac-diff system, on the current diffusion solver. 

   .. py:attribute:: numAllVoxels

      unsigned int (*value field*) Number of voxels in the core reac-diff system, on the current diffusion solver. 

   .. py:attribute:: numPools

      unsigned int (*value field*) Number of molecular pools in the entire reac-diff system, including variable, function and buffered.

   .. py:attribute:: nVec

      unsigned int,vector<double> (*lookup field*) vector of # of molecules along diffusion length, looked up by pool index

.. py:class:: Enz

.. py:class:: EnzBase

   Abstract base class for enzymes.

   .. py:attribute:: sub

      void (*shared message field*) Connects to substrate molecule

   .. py:attribute:: prd

      void (*shared message field*) Connects to product molecule

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setKm

       (*destination message field*) Assigns field value.

   .. py:method:: getKm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumKm

       (*destination message field*) Assigns field value.

   .. py:method:: getNumKm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setKcat

       (*destination message field*) Assigns field value.

   .. py:method:: getKcat

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumSubstrates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: enzDest

       (*destination message field*) Handles # of molecules of Enzyme

   .. py:method:: subDest

       (*destination message field*) Handles # of molecules of substrate

   .. py:method:: prdDest

       (*destination message field*) Handles # of molecules of product. Dummy.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: remesh

       (*destination message field*) Tells the MMEnz to recompute its numKm after remeshing

   .. py:attribute:: subOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: prdOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: Km

      double (*value field*) Michaelis-Menten constant in SI conc units (milliMolar)

   .. py:attribute:: numKm

      double (*value field*) Michaelis-Menten constant in number units, volume dependent

   .. py:attribute:: kcat

      double (*value field*) Forward rate constant for enzyme, units 1/sec

   .. py:attribute:: numSubstrates

      unsigned int (*value field*) Number of substrates in this MM reaction. Usually 1.Does not include the enzyme itself

.. py:class:: Finfo

   .. py:method:: getFieldName

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDocs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getType

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSrc

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDest

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: fieldName

      string (*value field*) Name of field handled by Finfo

   .. py:attribute:: docs

      string (*value field*) Documentation for Finfo

   .. py:attribute:: type

      string (*value field*) RTTI type info for this Finfo

   .. py:attribute:: src

      vector<string> (*value field*) Subsidiary SrcFinfos. Useful for SharedFinfos

   .. py:attribute:: dest

      vector<string> (*value field*) Subsidiary DestFinfos. Useful for SharedFinfos

.. py:class:: Func

   Func: general purpose function calculator using real numbers. It can

   parse mathematical expression defining a function and evaluate it

   and/or its derivative for specified variable values.

   The variables can be input from other moose objects. In case of

   arbitrary variable names, the source message must have the variable

   name as the first argument. For most common cases, input messages to

   set x, y, z and xy, xyz are made available without such

   requirement. This class handles only real numbers

    pi=3.141592...,

   e=2.718281... 

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: getValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDerivative

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMode

       (*destination message field*) Assigns field value.

   .. py:method:: getMode

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setExpr

       (*destination message field*) Assigns field value.

   .. py:method:: getExpr

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVar

       (*destination message field*) Assigns field value.

   .. py:method:: getVar

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getVars

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: varIn

       (*destination message field*) Handle value for specified variable coming from other objects

   .. py:method:: xIn

       (*destination message field*) Handle value for variable named x. This is a shorthand. If the
expression does not have any variable named x, this the first variable
in the sequence `vars`.

   .. py:method:: yIn

       (*destination message field*) Handle value for variable named y. This is a utility for two/three
 variable functions where the y value comes from a source separate
 from that of x. This is a shorthand. If the
expression does not have any variable named y, this the second
variable in the sequence `vars`.

   .. py:method:: zIn

       (*destination message field*) Handle value for variable named z. This is a utility for three
 variable functions where the z value comes from a source separate
 from that of x or y. This is a shorthand. If the expression does not
 have any variable named y, this the second variable in the sequence `vars`.

   .. py:method:: xyIn

       (*destination message field*) Handle value for variables x and y for two-variable function

   .. py:method:: xyzIn

       (*destination message field*) Handle value for variables x, y and z for three-variable function

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: valueOut

      double (*source message field*) Evaluated value of the function for the current variable values.

   .. py:attribute:: derivativeOut

      double (*source message field*) Value of derivative of the function for the current variable values

   .. py:attribute:: value

      double (*value field*) Result of the function evaluation with current variable values.

   .. py:attribute:: derivative

      double (*value field*) Derivative of the function at given variable values.

   .. py:attribute:: mode

      unsigned int (*value field*) Mode of operation: 
 1: only the function value will be funculated
 2: only the derivative will be funculated
 3: both function value and derivative at current variable values will be funculated.

   .. py:attribute:: expr

      string (*value field*) Mathematical expression defining the function. The underlying parser
is muParser. Hence the available functions and operators are (from
muParser docs):

Functions
Name        args    explanation
sin         1       sine function
cos         1       cosine function
tan         1       tangens function
asin        1       arcus sine function
acos        1       arcus cosine function
atan        1       arcus tangens function
sinh        1       hyperbolic sine function
cosh        1       hyperbolic cosine
tanh        1       hyperbolic tangens function
asinh       1       hyperbolic arcus sine function
acosh       1       hyperbolic arcus tangens function
atanh       1       hyperbolic arcur tangens function
log2        1       logarithm to the base 2
log10       1       logarithm to the base 10
log         1       logarithm to the base 10
ln  1       logarithm to base e (2.71828...)
exp         1       e raised to the power of x
sqrt        1       square root of a value
sign        1       sign function -1 if x<0; 1 if x>0
rint        1       round to nearest integer
abs         1       absolute value
min         var.    min of all arguments
max         var.    max of all arguments
sum         var.    sum of all arguments
avg         var.    mean value of all arguments

Operators
Op  meaning         prioroty
=   assignement     -1
&&  logical and     1
||  logical or      2
<=  less or equal   4
>=  greater or equal        4
!=  not equal       4
==  equal   4
>   greater than    4
<   less than       4
+   addition        5
-   subtraction     5
*   multiplication  6
/   division        6
^   raise x to the power of y       7

?:  if then else operator   C++ style syntax


   .. py:attribute:: vars

      vector<string> (*value field*) Variable names in the expression

   .. py:attribute:: x

      double (*value field*) Value for variable named x. This is a shorthand. If the
expression does not have any variable named x, this the first variable
in the sequence `vars`.

   .. py:attribute:: y

      double (*value field*) Value for variable named y. This is a utility for two/three
 variable functions where the y value comes from a source separate
 from that of x. This is a shorthand. If the
expression does not have any variable named y, this the second
variable in the sequence `vars`.

   .. py:attribute:: z

      double (*value field*) Value for variable named z. This is a utility for three
 variable functions where the z value comes from a source separate
 from that of x or z. This is a shorthand. If the expression does not
 have any variable named z, this the third variable in the sequence `vars`.

   .. py:attribute:: var

      string,double (*lookup field*) Lookup table for variable values.

.. py:class:: FuncBase

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: getResult

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: input

       (*destination message field*) Handles input values. This generic message works only in cases where the inputs  are commutative, so ordering does not matter.  In due course will implement a synapse type extendable,  identified system of inputs so that arbitrary numbers of  inputs can be unambiguaously defined. 

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: output

      double (*source message field*) Sends out sum on each timestep

   .. py:attribute:: result

      double (*value field*) Outcome of function computation

.. py:class:: FuncPool

   .. py:method:: input

       (*destination message field*) Handles input to control value of n_

.. py:class:: GapJunction

   Implementation of gap junction between two compartments. The shared

   fields, 'channel1' and 'channel2' can be connected to the 'channel'

   message of the compartments at either end of the gap junction. The

   compartments will send their Vm to the gap junction and receive the

   conductance 'Gk' of the gap junction and the Vm of the other

   compartment.

   .. py:attribute:: channel1

      void (*shared message field*) This is a shared message to couple the conductance and Vm from
terminal 2 to the compartment at terminal 1. The first entry is source
sending out Gk and Vm2, the second entry is destination for Vm1.

   .. py:attribute:: channel2

      void (*shared message field*) This is a shared message to couple the conductance and Vm from
terminal 1 to the compartment at terminal 2. The first entry is source
sending out Gk and Vm1, the second entry is destination for Vm2.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects. The Process should be called _second_ in each clock tick, after the Init message.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: Vm1

       (*destination message field*) Handles Vm message from compartment

   .. py:method:: Vm2

       (*destination message field*) Handles Vm message from another compartment

   .. py:method:: setGk

       (*destination message field*) Assigns field value.

   .. py:method:: getGk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles 'process' call

   .. py:method:: reinit

       (*destination message field*) Handles 'reinit' call

   .. py:attribute:: channel1Out

      double,double (*source message field*) Sends Gk and Vm from one compartment to the other

   .. py:attribute:: channel2Out

      double,double (*source message field*) Sends Gk and Vm from one compartment to the other

   .. py:attribute:: Gk

      double (*value field*) Conductance of the gap junction

.. py:class:: Group

   .. py:attribute:: group

      void (*source message field*) Handle for grouping Elements

.. py:class:: Gsolve

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setStoich

       (*destination message field*) Assigns field value.

   .. py:method:: getStoich

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumLocalVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNVec

       (*destination message field*) Assigns field value.

   .. py:method:: getNVec

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumAllVoxels

       (*destination message field*) Assigns field value.

   .. py:method:: getNumAllVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumPools

       (*destination message field*) Assigns field value.

   .. py:method:: getNumPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setUseRandInit

       (*destination message field*) Assigns field value.

   .. py:method:: getUseRandInit

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: stoich

      Id (*value field*) Stoichiometry object for handling this reaction system.

   .. py:attribute:: numLocalVoxels

      unsigned int (*value field*) Number of voxels in the core reac-diff system, on the current solver. 

   .. py:attribute:: numAllVoxels

      unsigned int (*value field*) Number of voxels in the entire reac-diff system, including proxy voxels to represent abutting compartments.

   .. py:attribute:: numPools

      unsigned int (*value field*) Number of molecular pools in the entire reac-diff system, including variable, function and buffered.

   .. py:attribute:: useRandInit

      bool (*value field*) Flag: True when using probabilistic (random) rounding. When initializing the mol# from floating-point Sinit values, we have two options. One is to look at each Sinit, and round to the nearest integer. The other is to look at each Sinit, and probabilistically round up or down depending on the  value. For example, if we had a Sinit value of 1.49,  this would always be rounded to 1.0 if the flag is false, and would be rounded to 1.0 and 2.0 in the ratio 51:49 if the flag is true. 

   .. py:attribute:: nVec

      unsigned int,vector<double> (*lookup field*) vector of pool counts

.. py:class:: HHChannel

   HHChannel: Hodgkin-Huxley type voltage-gated Ion channel. Something like the old tabchannel from GENESIS, but also presents a similar interface as hhchan from GENESIS. 

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on.
 The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setXpower

       (*destination message field*) Assigns field value.

   .. py:method:: getXpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYpower

       (*destination message field*) Assigns field value.

   .. py:method:: getYpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZpower

       (*destination message field*) Assigns field value.

   .. py:method:: getZpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInstant

       (*destination message field*) Assigns field value.

   .. py:method:: getInstant

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setUseConcentration

       (*destination message field*) Assigns field value.

   .. py:method:: getUseConcentration

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: concen

       (*destination message field*) Incoming message from Concen object to specific conc to usein the Z gate calculations

   .. py:method:: createGate

       (*destination message field*) Function to create specified gate.Argument: Gate type [X Y Z]

   .. py:method:: setNumGateX

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateX

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateY

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateY

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateZ

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateZ

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:attribute:: Xpower

      double (*value field*) Power for X gate

   .. py:attribute:: Ypower

      double (*value field*) Power for Y gate

   .. py:attribute:: Zpower

      double (*value field*) Power for Z gate

   .. py:attribute:: instant

      int (*value field*) Bitmapped flag: bit 0 = Xgate, bit 1 = Ygate, bit 2 = ZgateWhen true, specifies that the lookup table value should beused directly as the state of the channel, rather than usedas a rate term for numerical integration for the state

   .. py:attribute:: X

      double (*value field*) State variable for X gate

   .. py:attribute:: Y

      double (*value field*) State variable for Y gate

   .. py:attribute:: Z

      double (*value field*) State variable for Y gate

   .. py:attribute:: useConcentration

      int (*value field*) Flag: when true, use concentration message rather than Vm tocontrol Z gate

.. py:class:: HHChannel2D

   HHChannel2D: Hodgkin-Huxley type voltage-gated Ion channel. Something like the old tabchannel from GENESIS, but also presents a similar interface as hhchan from GENESIS. 

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on.
 The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setXindex

       (*destination message field*) Assigns field value.

   .. py:method:: getXindex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYindex

       (*destination message field*) Assigns field value.

   .. py:method:: getYindex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZindex

       (*destination message field*) Assigns field value.

   .. py:method:: getZindex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXpower

       (*destination message field*) Assigns field value.

   .. py:method:: getXpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYpower

       (*destination message field*) Assigns field value.

   .. py:method:: getYpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZpower

       (*destination message field*) Assigns field value.

   .. py:method:: getZpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInstant

       (*destination message field*) Assigns field value.

   .. py:method:: getInstant

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: concen

       (*destination message field*) Incoming message from Concen object to specific conc to useas the first concen variable

   .. py:method:: concen2

       (*destination message field*) Incoming message from Concen object to specific conc to useas the second concen variable

   .. py:method:: setNumGateX

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateX

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateY

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateY

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateZ

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateZ

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:attribute:: Xindex

      string (*value field*) String for setting X index.

   .. py:attribute:: Yindex

      string (*value field*) String for setting Y index.

   .. py:attribute:: Zindex

      string (*value field*) String for setting Z index.

   .. py:attribute:: Xpower

      double (*value field*) Power for X gate

   .. py:attribute:: Ypower

      double (*value field*) Power for Y gate

   .. py:attribute:: Zpower

      double (*value field*) Power for Z gate

   .. py:attribute:: instant

      int (*value field*) Bitmapped flag: bit 0 = Xgate, bit 1 = Ygate, bit 2 = ZgateWhen true, specifies that the lookup table value should beused directly as the state of the channel, rather than usedas a rate term for numerical integration for the state

   .. py:attribute:: X

      double (*value field*) State variable for X gate

   .. py:attribute:: Y

      double (*value field*) State variable for Y gate

   .. py:attribute:: Z

      double (*value field*) State variable for Y gate

.. py:class:: HHGate

   HHGate: Gate for Hodkgin-Huxley type channels, equivalent to the m and h terms on the Na squid channel and the n term on K. This takes the voltage and state variable from the channel, computes the new value of the state variable and a scaling, depending on gate power, for the conductance.

   .. py:method:: getA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAlpha

       (*destination message field*) Assigns field value.

   .. py:method:: getAlpha

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBeta

       (*destination message field*) Assigns field value.

   .. py:method:: getBeta

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau

       (*destination message field*) Assigns field value.

   .. py:method:: getTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMInfinity

       (*destination message field*) Assigns field value.

   .. py:method:: getMInfinity

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMin

       (*destination message field*) Assigns field value.

   .. py:method:: getMin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMax

       (*destination message field*) Assigns field value.

   .. py:method:: getMax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDivs

       (*destination message field*) Assigns field value.

   .. py:method:: getDivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTableA

       (*destination message field*) Assigns field value.

   .. py:method:: getTableA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTableB

       (*destination message field*) Assigns field value.

   .. py:method:: getTableB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setUseInterpolation

       (*destination message field*) Assigns field value.

   .. py:method:: getUseInterpolation

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAlphaParms

       (*destination message field*) Assigns field value.

   .. py:method:: getAlphaParms

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setupAlpha

       (*destination message field*) Set up both gates using 13 parameters, as follows:setupAlpha AA AB AC AD AF BA BB BC BD BF xdivs xmin xmaxHere AA-AF are Coefficients A to F of the alpha (forward) termHere BA-BF are Coefficients A to F of the beta (reverse) termHere xdivs is the number of entries in the table,xmin and xmax define the range for lookup.Outside this range the returned value will be the low [high]entry of the table.The equation describing each table is:y(x) = (A + B * x) / (C + exp((x + D) / F))The original HH equations can readily be cast into this form

   .. py:method:: setupTau

       (*destination message field*) Identical to setupAlpha, except that the forms specified bythe 13 parameters are for the tau and m-infinity curves ratherthan the alpha and beta terms. So the parameters are:setupTau TA TB TC TD TF MA MB MC MD MF xdivs xmin xmaxAs before, the equation describing each curve is:y(x) = (A + B * x) / (C + exp((x + D) / F))

   .. py:method:: tweakAlpha

       (*destination message field*) Dummy function for backward compatibility. It used to convertthe tables from alpha, beta values to alpha, alpha+betabecause the internal calculations used these forms. Notneeded now, deprecated.

   .. py:method:: tweakTau

       (*destination message field*) Dummy function for backward compatibility. It used to convertthe tables from tau, minf values to alpha, alpha+betabecause the internal calculations used these forms. Notneeded now, deprecated.

   .. py:method:: setupGate

       (*destination message field*) Sets up one gate at a time using the alpha/beta form.Has 9 parameters, as follows:setupGate A B C D F xdivs xmin xmax is_betaThis sets up the gate using the equation:y(x) = (A + B * x) / (C + exp((x + D) / F))Deprecated.

   .. py:attribute:: alpha

      vector<double> (*value field*) Parameters for voltage-dependent rates, alpha:Set up alpha term using 5 parameters, as follows:y(x) = (A + B * x) / (C + exp((x + D) / F))The original HH equations can readily be cast into this form

   .. py:attribute:: beta

      vector<double> (*value field*) Parameters for voltage-dependent rates, beta:Set up beta term using 5 parameters, as follows:y(x) = (A + B * x) / (C + exp((x + D) / F))The original HH equations can readily be cast into this form

   .. py:attribute:: tau

      vector<double> (*value field*) Parameters for voltage-dependent rates, tau:Set up tau curve using 5 parameters, as follows:y(x) = (A + B * x) / (C + exp((x + D) / F))

   .. py:attribute:: mInfinity

      vector<double> (*value field*) Parameters for voltage-dependent rates, mInfinity:Set up mInfinity curve using 5 parameters, as follows:y(x) = (A + B * x) / (C + exp((x + D) / F))The original HH equations can readily be cast into this form

   .. py:attribute:: min

      double (*value field*) Minimum range for lookup

   .. py:attribute:: max

      double (*value field*) Minimum range for lookup

   .. py:attribute:: divs

      unsigned int (*value field*) Divisions for lookup. Zero means to use linear interpolation

   .. py:attribute:: tableA

      vector<double> (*value field*) Table of A entries

   .. py:attribute:: tableB

      vector<double> (*value field*) Table of alpha + beta entries

   .. py:attribute:: useInterpolation

      bool (*value field*) Flag: use linear interpolation if true, else direct lookup

   .. py:attribute:: alphaParms

      vector<double> (*value field*) Set up both gates using 13 parameters, as follows:setupAlpha AA AB AC AD AF BA BB BC BD BF xdivs xmin xmaxHere AA-AF are Coefficients A to F of the alpha (forward) termHere BA-BF are Coefficients A to F of the beta (reverse) termHere xdivs is the number of entries in the table,xmin and xmax define the range for lookup.Outside this range the returned value will be the low [high]entry of the table.The equation describing each table is:y(x) = (A + B * x) / (C + exp((x + D) / F))The original HH equations can readily be cast into this form

   .. py:attribute:: A

      double,double (*lookup field*) lookupA: Look up the A gate value from a double. Usually doesso by direct scaling and offset to an integer lookup, usinga fine enough table granularity that there is little error.Alternatively uses linear interpolation.The range of the double is predefined based on knowledge ofvoltage or conc ranges, and the granularity is specified bythe xmin, xmax, and dV fields.

   .. py:attribute:: B

      double,double (*lookup field*) lookupB: Look up the B gate value from a double.Note that this looks up the raw tables, which are transformedfrom the reference parameters.

.. py:class:: HHGate2D

   HHGate2D: Gate for Hodkgin-Huxley type channels, equivalent to the m and h terms on the Na squid channel and the n term on K. This takes the voltage and state variable from the channel, computes the new value of the state variable and a scaling, depending on gate power, for the conductance. These two terms are sent right back in a message to the channel.

   .. py:method:: getA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTableA

       (*destination message field*) Assigns field value.

   .. py:method:: getTableA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTableB

       (*destination message field*) Assigns field value.

   .. py:method:: getTableB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXminA

       (*destination message field*) Assigns field value.

   .. py:method:: getXminA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmaxA

       (*destination message field*) Assigns field value.

   .. py:method:: getXmaxA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXdivsA

       (*destination message field*) Assigns field value.

   .. py:method:: getXdivsA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYminA

       (*destination message field*) Assigns field value.

   .. py:method:: getYminA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmaxA

       (*destination message field*) Assigns field value.

   .. py:method:: getYmaxA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYdivsA

       (*destination message field*) Assigns field value.

   .. py:method:: getYdivsA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXminB

       (*destination message field*) Assigns field value.

   .. py:method:: getXminB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmaxB

       (*destination message field*) Assigns field value.

   .. py:method:: getXmaxB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXdivsB

       (*destination message field*) Assigns field value.

   .. py:method:: getXdivsB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYminB

       (*destination message field*) Assigns field value.

   .. py:method:: getYminB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmaxB

       (*destination message field*) Assigns field value.

   .. py:method:: getYmaxB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYdivsB

       (*destination message field*) Assigns field value.

   .. py:method:: getYdivsB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: tableA

      vector< vector<double> > (*value field*) Table of A entries

   .. py:attribute:: tableB

      vector< vector<double> > (*value field*) Table of B entries

   .. py:attribute:: xminA

      double (*value field*) Minimum range for lookup

   .. py:attribute:: xmaxA

      double (*value field*) Minimum range for lookup

   .. py:attribute:: xdivsA

      unsigned int (*value field*) Divisions for lookup. Zero means to use linear interpolation

   .. py:attribute:: yminA

      double (*value field*) Minimum range for lookup

   .. py:attribute:: ymaxA

      double (*value field*) Minimum range for lookup

   .. py:attribute:: ydivsA

      unsigned int (*value field*) Divisions for lookup. Zero means to use linear interpolation

   .. py:attribute:: xminB

      double (*value field*) Minimum range for lookup

   .. py:attribute:: xmaxB

      double (*value field*) Minimum range for lookup

   .. py:attribute:: xdivsB

      unsigned int (*value field*) Divisions for lookup. Zero means to use linear interpolation

   .. py:attribute:: yminB

      double (*value field*) Minimum range for lookup

   .. py:attribute:: ymaxB

      double (*value field*) Minimum range for lookup

   .. py:attribute:: ydivsB

      unsigned int (*value field*) Divisions for lookup. Zero means to use linear interpolation

   .. py:attribute:: A

      vector<double>,double (*lookup field*) lookupA: Look up the A gate value from two doubles, passedin as a vector. Uses linear interpolation in the 2D tableThe range of the lookup doubles is predefined based on knowledge of voltage or conc ranges, and the granularity is specified by the xmin, xmax, and dx field, and their y-axis counterparts.

   .. py:attribute:: B

      vector<double>,double (*lookup field*) lookupB: Look up B gate value from two doubles in a vector.

.. py:class:: HSolve

   .. py:attribute:: proc

      void (*shared message field*) Handles 'reinit' and 'process' calls from a clock.

   .. py:method:: setSeed

       (*destination message field*) Assigns field value.

   .. py:method:: getSeed

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTarget

       (*destination message field*) Assigns field value.

   .. py:method:: getTarget

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDt

       (*destination message field*) Assigns field value.

   .. py:method:: getDt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaAdvance

       (*destination message field*) Assigns field value.

   .. py:method:: getCaAdvance

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVDiv

       (*destination message field*) Assigns field value.

   .. py:method:: getVDiv

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVMin

       (*destination message field*) Assigns field value.

   .. py:method:: getVMin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVMax

       (*destination message field*) Assigns field value.

   .. py:method:: getVMax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaDiv

       (*destination message field*) Assigns field value.

   .. py:method:: getCaDiv

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaMin

       (*destination message field*) Assigns field value.

   .. py:method:: getCaMin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaMax

       (*destination message field*) Assigns field value.

   .. py:method:: getCaMax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles 'process' call: Solver advances by one time-step.

   .. py:method:: reinit

       (*destination message field*) Handles 'reinit' call: Solver reads in model.

   .. py:attribute:: seed

      Id (*value field*) Use this field to specify path to a 'seed' compartment, that is, any compartment within a neuron. The HSolve object uses this seed as a handle to discover the rest of the neuronal model, which means all the remaining compartments, channels, synapses, etc.

   .. py:attribute:: target

      string (*value field*) Specifies the path to a compartmental model to be taken over. This can be the path to any container object that has the model under it (found by performing a deep search). Alternatively, this can also be the path to any compartment within the neuron. This compartment will be used as a handle to discover the rest of the model, which means all the remaining compartments, channels, synapses, etc.

   .. py:attribute:: dt

      double (*value field*) The time-step for this solver.

   .. py:attribute:: caAdvance

      int (*value field*) This flag determines how current flowing into a calcium pool is computed. A value of 0 means that the membrane potential at the beginning of the time-step is used for the calculation. This is how GENESIS does its computations. A value of 1 means the membrane potential at the middle of the time-step is used. This is the correct way of integration, and is the default way.

   .. py:attribute:: vDiv

      int (*value field*) Specifies number of divisions for lookup tables of voltage-sensitive channels.

   .. py:attribute:: vMin

      double (*value field*) Specifies the lower bound for lookup tables of voltage-sensitive channels. Default is to automatically decide based on the tables of the channels that the solver reads in.

   .. py:attribute:: vMax

      double (*value field*) Specifies the upper bound for lookup tables of voltage-sensitive channels. Default is to automatically decide based on the tables of the channels that the solver reads in.

   .. py:attribute:: caDiv

      int (*value field*) Specifies number of divisions for lookup tables of calcium-sensitive channels.

   .. py:attribute:: caMin

      double (*value field*) Specifies the lower bound for lookup tables of calcium-sensitive channels. Default is to automatically decide based on the tables of the channels that the solver reads in.

   .. py:attribute:: caMax

      double (*value field*) Specifies the upper bound for lookup tables of calcium-sensitive channels. Default is to automatically decide based on the tables of the channels that the solver reads in.

.. py:class:: IntFire

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setVm

       (*destination message field*) Assigns field value.

   .. py:method:: getVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau

       (*destination message field*) Assigns field value.

   .. py:method:: getTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setThresh

       (*destination message field*) Assigns field value.

   .. py:method:: getThresh

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRefractoryPeriod

       (*destination message field*) Assigns field value.

   .. py:method:: getRefractoryPeriod

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBufferTime

       (*destination message field*) Assigns field value.

   .. py:method:: getBufferTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: spikeOut

      double (*source message field*) Sends out spike events

   .. py:attribute:: Vm

      double (*value field*) Membrane potential

   .. py:attribute:: tau

      double (*value field*) charging time-course

   .. py:attribute:: thresh

      double (*value field*) firing threshold

   .. py:attribute:: refractoryPeriod

      double (*value field*) Minimum time between successive spikes

   .. py:attribute:: bufferTime

      double (*value field*) Duration of spike buffer.

.. py:class:: Interpol

   Interpol: Interpolation class. Handles lookup from a 1-dimensional array of real-numbered values.Returns 'y' value based on given 'x' value. Can either use interpolation or roundoff to the nearest index.

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setXmin

       (*destination message field*) Assigns field value.

   .. py:method:: getXmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmax

       (*destination message field*) Assigns field value.

   .. py:method:: getXmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: input

       (*destination message field*) Interpolates using the input as x value.

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: lookupOut

      double (*source message field*) respond to a request for a value lookup

   .. py:attribute:: xmin

      double (*value field*) Minimum value of x. x below this will result in y[0] being returned.

   .. py:attribute:: xmax

      double (*value field*) Maximum value of x. x above this will result in y[last] being returned.

   .. py:attribute:: y

      double (*value field*) Looked up value.

.. py:class:: Interpol2D

   Interpol2D: Interpolation class. Handles lookup from a 2-dimensional grid of real-numbered values. Returns 'z' value based on given 'x' and 'y' values. Can either use interpolation or roundoff to the nearest index.

   .. py:attribute:: lookupReturn2D

      void (*shared message field*) This is a shared message for doing lookups on the table. Receives 2 doubles: x, y. Sends back a double with the looked-up z value.

   .. py:method:: lookup

       (*destination message field*) Looks up table value based on indices v1 and v2, and sendsvalue back using the 'lookupOut' message

   .. py:method:: setXmin

       (*destination message field*) Assigns field value.

   .. py:method:: getXmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmax

       (*destination message field*) Assigns field value.

   .. py:method:: getXmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXdivs

       (*destination message field*) Assigns field value.

   .. py:method:: getXdivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDx

       (*destination message field*) Assigns field value.

   .. py:method:: getDx

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmin

       (*destination message field*) Assigns field value.

   .. py:method:: getYmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmax

       (*destination message field*) Assigns field value.

   .. py:method:: getYmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYdivs

       (*destination message field*) Assigns field value.

   .. py:method:: getYdivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDy

       (*destination message field*) Assigns field value.

   .. py:method:: getDy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTable

       (*destination message field*) Assigns field value.

   .. py:method:: getTable

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTableVector2D

       (*destination message field*) Assigns field value.

   .. py:method:: getTableVector2D

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: lookupOut

      double (*source message field*) respond to a request for a value lookup

   .. py:attribute:: xmin

      double (*value field*) Minimum value for x axis of lookup table

   .. py:attribute:: xmax

      double (*value field*) Maximum value for x axis of lookup table

   .. py:attribute:: xdivs

      unsigned int (*value field*) # of divisions on x axis of lookup table

   .. py:attribute:: dx

      double (*value field*) Increment on x axis of lookup table

   .. py:attribute:: ymin

      double (*value field*) Minimum value for y axis of lookup table

   .. py:attribute:: ymax

      double (*value field*) Maximum value for y axis of lookup table

   .. py:attribute:: ydivs

      unsigned int (*value field*) # of divisions on y axis of lookup table

   .. py:attribute:: dy

      double (*value field*) Increment on y axis of lookup table

   .. py:attribute:: tableVector2D

      vector< vector<double> > (*value field*) Get the entire table.

   .. py:attribute:: table

      vector<unsigned int>,double (*lookup field*) Lookup an entry on the table

   .. py:attribute:: z

      vector<double>,double (*lookup field*) Interpolated value for specified x and y. This is provided for debugging. Normally other objects will retrieve interpolated values via lookup message.

.. py:class:: IzhikevichNrn

   Izhikevich model of spiking neuron (Izhikevich,EM. 2003. Simple model of spiking neurons. Neural Networks, IEEE Transactions on 14(6). pp 1569-1572).

   

     dVm/dt = 0.04 * Vm^2 + 5 * Vm + 140 - u + inject

     du/dt = a * (b * Vm - u)

    if Vm >= Vmax then Vm = c and u = u + d

    Vmax = 30 mV in the paper.

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process message from scheduler

   .. py:attribute:: channel

      void (*shared message field*) This is a shared message from a IzhikevichNrn to channels.The first entry is a MsgDest for the info coming from the channel. It expects Gk and Ek from the channel as args. The second entry is a MsgSrc sending Vm 

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setVmax

       (*destination message field*) Assigns field value.

   .. py:method:: getVmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setC

       (*destination message field*) Assigns field value.

   .. py:method:: getC

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setD

       (*destination message field*) Assigns field value.

   .. py:method:: getD

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setA

       (*destination message field*) Assigns field value.

   .. py:method:: getA

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setB

       (*destination message field*) Assigns field value.

   .. py:method:: getB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getU

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVm

       (*destination message field*) Assigns field value.

   .. py:method:: getVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInject

       (*destination message field*) Assigns field value.

   .. py:method:: getInject

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRmByTau

       (*destination message field*) Assigns field value.

   .. py:method:: getRmByTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAccommodating

       (*destination message field*) Assigns field value.

   .. py:method:: getAccommodating

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setU0

       (*destination message field*) Assigns field value.

   .. py:method:: getU0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInitVm

       (*destination message field*) Assigns field value.

   .. py:method:: getInitVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInitU

       (*destination message field*) Assigns field value.

   .. py:method:: getInitU

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAlpha

       (*destination message field*) Assigns field value.

   .. py:method:: getAlpha

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBeta

       (*destination message field*) Assigns field value.

   .. py:method:: getBeta

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGamma

       (*destination message field*) Assigns field value.

   .. py:method:: getGamma

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: injectMsg

       (*destination message field*) Injection current into the neuron.

   .. py:method:: cDest

       (*destination message field*) Destination message to modify parameter c at runtime.

   .. py:method:: dDest

       (*destination message field*) Destination message to modify parameter d at runtime.

   .. py:method:: bDest

       (*destination message field*) Destination message to modify parameter b at runtime

   .. py:method:: aDest

       (*destination message field*) Destination message modify parameter a at runtime.

   .. py:method:: handleChannel

       (*destination message field*) Handles conductance and reversal potential arguments from Channel

   .. py:attribute:: VmOut

      double (*source message field*) Sends out Vm

   .. py:attribute:: spikeOut

      double (*source message field*) Sends out spike events

   .. py:attribute:: VmOut

      double (*source message field*) Sends out Vm

   .. py:attribute:: Vmax

      double (*value field*) Maximum membrane potential. Membrane potential is reset to c whenever it reaches Vmax. NOTE: Izhikevich model specifies the PEAK voltage, rather than THRSHOLD voltage. The threshold depends on the previous history.

   .. py:attribute:: c

      double (*value field*) Reset potential. Membrane potential is reset to c whenever it reaches Vmax.

   .. py:attribute:: d

      double (*value field*) Parameter d in Izhikevich model. Unit is V/s.

   .. py:attribute:: a

      double (*value field*) Parameter a in Izhikevich model. Unit is s^{-1}

   .. py:attribute:: b

      double (*value field*) Parameter b in Izhikevich model. Unit is s^{-1}

   .. py:attribute:: u

      double (*value field*) Parameter u in Izhikevich equation. Unit is V/s

   .. py:attribute:: Vm

      double (*value field*) Membrane potential, equivalent to v in Izhikevich equation.

   .. py:attribute:: Im

      double (*value field*) Total current going through the membrane. Unit is A.

   .. py:attribute:: inject

      double (*value field*) External current injection into the neuron

   .. py:attribute:: RmByTau

      double (*value field*) Hidden coefficient of input current term (I) in Izhikevich model. Defaults to 1e9 Ohm/s.

   .. py:attribute:: accommodating

      bool (*value field*) True if this neuron is an accommodating one. The equation for recovery variable u is special in this case.

   .. py:attribute:: u0

      double (*value field*) This is used for accommodating neurons where recovery variables u is computed as: u += tau*a*(b*(Vm-u0))

   .. py:attribute:: initVm

      double (*value field*) Initial membrane potential. Unit is V.

   .. py:attribute:: initU

      double (*value field*) Initial value of u.

   .. py:attribute:: alpha

      double (*value field*) Coefficient of v^2 in Izhikevich equation. Defaults to 0.04 in physiological unit. In SI it should be 40000.0. Unit is V^-1 s^{-1}

   .. py:attribute:: beta

      double (*value field*) Coefficient of v in Izhikevich model. Defaults to 5 in physiological unit, 5000.0 for SI units. Unit is s^{-1}

   .. py:attribute:: gamma

      double (*value field*) Constant term in Izhikevich model. Defaults to 140 in both physiological and SI units. unit is V/s.

.. py:class:: Ksolve

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setMethod

       (*destination message field*) Assigns field value.

   .. py:method:: getMethod

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEpsAbs

       (*destination message field*) Assigns field value.

   .. py:method:: getEpsAbs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEpsRel

       (*destination message field*) Assigns field value.

   .. py:method:: getEpsRel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setStoich

       (*destination message field*) Assigns field value.

   .. py:method:: getStoich

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDsolve

       (*destination message field*) Assigns field value.

   .. py:method:: getDsolve

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCompartment

       (*destination message field*) Assigns field value.

   .. py:method:: getCompartment

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumLocalVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNVec

       (*destination message field*) Assigns field value.

   .. py:method:: getNVec

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumAllVoxels

       (*destination message field*) Assigns field value.

   .. py:method:: getNumAllVoxels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumPools

       (*destination message field*) Assigns field value.

   .. py:method:: getNumPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: method

      string (*value field*) Integration method, using GSL. So far only explict. Options are:rk5: The default Runge-Kutta-Fehlberg 5th order adaptive dt methodgsl: alias for the aboverk4: The Runge-Kutta 4th order fixed dt methodrk2: The Runge-Kutta 2,3 embedded fixed dt methodrkck: The Runge-Kutta Cash-Karp (4,5) methodrk8: The Runge-Kutta Prince-Dormand (8,9) method

   .. py:attribute:: epsAbs

      double (*value field*) Absolute permissible integration error range.

   .. py:attribute:: epsRel

      double (*value field*) Relative permissible integration error range.

   .. py:attribute:: stoich

      Id (*value field*) Stoichiometry object for handling this reaction system.

   .. py:attribute:: dsolve

      Id (*value field*) Diffusion solver object handling this reactin system.

   .. py:attribute:: compartment

      Id (*value field*) Compartment in which the Ksolve reaction system lives.

   .. py:attribute:: numLocalVoxels

      unsigned int (*value field*) Number of voxels in the core reac-diff system, on the current solver. 

   .. py:attribute:: numAllVoxels

      unsigned int (*value field*) Number of voxels in the entire reac-diff system, including proxy voxels to represent abutting compartments.

   .. py:attribute:: numPools

      unsigned int (*value field*) Number of molecular pools in the entire reac-diff system, including variable, function and buffered.

   .. py:attribute:: nVec

      unsigned int,vector<double> (*lookup field*) vector of pool counts. Index specifies which voxel.

.. py:class:: Leakage

   Leakage: Passive leakage channel.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from the scheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on.
The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

.. py:class:: Long

   Variable for storing values.

   .. py:method:: setValue

       (*destination message field*) Assigns field value.

   .. py:method:: getValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: value

      long (*value field*) Variable value

.. py:class:: MMenz

.. py:class:: MarkovChannel

   MarkovChannel : Multistate ion channel class.It deals with ion channels which can be found in one of multiple states, some of which are conducting. This implementation assumes the occurence of first order kinetics to calculate the probabilities of the channel being found in all states. Further, the rates of transition between these states can be constant, voltage-dependent or ligand dependent (only one ligand species). The current flow obtained from the channel is calculated in a deterministic method by solving the system of differential equations obtained from the assumptions above.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setLigandConc

       (*destination message field*) Assigns field value.

   .. py:method:: getLigandConc

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVm

       (*destination message field*) Assigns field value.

   .. py:method:: getVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumStates

       (*destination message field*) Assigns field value.

   .. py:method:: getNumStates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumOpenStates

       (*destination message field*) Assigns field value.

   .. py:method:: getNumOpenStates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInitialState

       (*destination message field*) Assigns field value.

   .. py:method:: getInitialState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLabels

       (*destination message field*) Assigns field value.

   .. py:method:: getLabels

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGbar

       (*destination message field*) Assigns field value.

   .. py:method:: getGbar

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: handleLigandConc

       (*destination message field*) Deals with incoming messages containing information of ligand concentration

   .. py:method:: handleState

       (*destination message field*) Deals with incoming message from MarkovSolver object containing state information of the channel.


   .. py:attribute:: ligandConc

      double (*value field*) Ligand concentration.

   .. py:attribute:: Vm

      double (*value field*) Membrane voltage.

   .. py:attribute:: numStates

      unsigned int (*value field*) The number of states that the channel can occupy.

   .. py:attribute:: numOpenStates

      unsigned int (*value field*) The number of states which are open/conducting.

   .. py:attribute:: state

      vector<double> (*value field*) This is a row vector that contains the probabilities of finding the channel in each state.

   .. py:attribute:: initialState

      vector<double> (*value field*) This is a row vector that contains the probabilities of finding the channel in each state at t = 0. The state of the channel is reset to this value during a call to reinit()

   .. py:attribute:: labels

      vector<string> (*value field*) Labels for each state.

   .. py:attribute:: gbar

      vector<double> (*value field*) A row vector containing the conductance associated with each of the open/conducting states.

.. py:class:: MarkovGslSolver

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: getIsInitialized

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMethod

       (*destination message field*) Assigns field value.

   .. py:method:: getMethod

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRelativeAccuracy

       (*destination message field*) Assigns field value.

   .. py:method:: getRelativeAccuracy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAbsoluteAccuracy

       (*destination message field*) Assigns field value.

   .. py:method:: getAbsoluteAccuracy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInternalDt

       (*destination message field*) Assigns field value.

   .. py:method:: getInternalDt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: init

       (*destination message field*) Initialize solver parameters.

   .. py:method:: handleQ

       (*destination message field*) Handles information regarding the instantaneous rate matrix from the MarkovRateTable class.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: stateOut

      vector<double> (*source message field*) Sends updated state to the MarkovChannel class.

   .. py:attribute:: isInitialized

      bool (*value field*) True if the message has come in to set solver parameters.

   .. py:attribute:: method

      string (*value field*) Numerical method to use.

   .. py:attribute:: relativeAccuracy

      double (*value field*) Accuracy criterion

   .. py:attribute:: absoluteAccuracy

      double (*value field*) Another accuracy criterion

   .. py:attribute:: internalDt

      double (*value field*) internal timestep to use.

.. py:class:: MarkovRateTable

   .. py:attribute:: channel

      void (*shared message field*) This message couples the rate table to the compartment. The rate table needs updates on voltage in order to compute the rate table.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: handleVm

       (*destination message field*) Handles incoming message containing voltage information.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: init

       (*destination message field*) Initialization of the class. Allocates memory for all the tables.

   .. py:method:: handleLigandConc

       (*destination message field*) Handles incoming message containing ligand concentration.

   .. py:method:: set1d

       (*destination message field*) Setting up of 1D lookup table for the (i,j)'th rate.

   .. py:method:: set2d

       (*destination message field*) Setting up of 2D lookup table for the (i,j)'th rate.

   .. py:method:: setconst

       (*destination message field*) Setting a constant value for the (i,j)'th rate. Internally, this is	stored as a 1-D rate with a lookup table containing 1 entry.

   .. py:method:: setVm

       (*destination message field*) Assigns field value.

   .. py:method:: getVm

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLigandConc

       (*destination message field*) Assigns field value.

   .. py:method:: getLigandConc

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getQ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSize

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: instratesOut

      vector< vector<double> > (*source message field*) Sends out instantaneous rate information of varying transition ratesat each time step.

   .. py:attribute:: Vm

      double (*value field*) Membrane voltage.

   .. py:attribute:: ligandConc

      double (*value field*) Ligand concentration.

   .. py:attribute:: Q

      vector< vector<double> > (*value field*) Instantaneous rate matrix.

   .. py:attribute:: size

      unsigned int (*value field*) Dimension of the families of lookup tables. Is always equal to the number of states in the model.

.. py:class:: MarkovSolver

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

.. py:class:: MarkovSolverBase

   .. py:attribute:: channel

      void (*shared message field*) This message couples the MarkovSolverBase to the Compartment. The compartment needs Vm in order to look up the correct matrix exponential for computing the state.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: handleVm

       (*destination message field*) Handles incoming message containing voltage information.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: ligandConc

       (*destination message field*) Handles incoming message containing ligand concentration.

   .. py:method:: init

       (*destination message field*) Setups the table of matrix exponentials associated with the solver object.

   .. py:method:: getQ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInitialState

       (*destination message field*) Assigns field value.

   .. py:method:: getInitialState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmin

       (*destination message field*) Assigns field value.

   .. py:method:: getXmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmax

       (*destination message field*) Assigns field value.

   .. py:method:: getXmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXdivs

       (*destination message field*) Assigns field value.

   .. py:method:: getXdivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getInvdx

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmin

       (*destination message field*) Assigns field value.

   .. py:method:: getYmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYmax

       (*destination message field*) Assigns field value.

   .. py:method:: getYmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYdivs

       (*destination message field*) Assigns field value.

   .. py:method:: getYdivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getInvdy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: stateOut

      vector<double> (*source message field*) Sends updated state to the MarkovChannel class.

   .. py:attribute:: Q

      vector< vector<double> > (*value field*) Instantaneous rate matrix.

   .. py:attribute:: state

      vector<double> (*value field*) Current state of the channel.

   .. py:attribute:: initialState

      vector<double> (*value field*) Initial state of the channel.

   .. py:attribute:: xmin

      double (*value field*) Minimum value for x axis of lookup table

   .. py:attribute:: xmax

      double (*value field*) Maximum value for x axis of lookup table

   .. py:attribute:: xdivs

      unsigned int (*value field*) # of divisions on x axis of lookup table

   .. py:attribute:: invdx

      double (*value field*) Reciprocal of increment on x axis of lookup table

   .. py:attribute:: ymin

      double (*value field*) Minimum value for y axis of lookup table

   .. py:attribute:: ymax

      double (*value field*) Maximum value for y axis of lookup table

   .. py:attribute:: ydivs

      unsigned int (*value field*) # of divisions on y axis of lookup table

   .. py:attribute:: invdy

      double (*value field*) Reciprocal of increment on y axis of lookup table

.. py:class:: MathFunc

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setMathML

       (*destination message field*) Assigns field value.

   .. py:method:: getMathML

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFunction

       (*destination message field*) Assigns field value.

   .. py:method:: getFunction

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getResult

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: arg1

       (*destination message field*) Handle arg1

   .. py:method:: arg2

       (*destination message field*) Handle arg2

   .. py:method:: arg3

       (*destination message field*) Handle arg3

   .. py:method:: arg4

       (*destination message field*) Handle arg4

   .. py:method:: process

       (*destination message field*) Handle process call

   .. py:method:: reinit

       (*destination message field*) Handle reinit call

   .. py:attribute:: output

      double (*source message field*) Sends out result of computation

   .. py:attribute:: mathML

      string (*value field*) MathML version of expression to compute

   .. py:attribute:: function

      string (*value field*) function is for functions of form f(x, y) = x + y

   .. py:attribute:: result

      double (*value field*) result value

.. py:class:: MeshEntry

   One voxel in a chemical reaction compartment

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:attribute:: mesh

      void (*shared message field*) Shared message for updating mesh volumes and subdivisions,typically controls pool volumes

   .. py:method:: getVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDimensions

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMeshType

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getCoordinates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNeighbors

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDiffusionArea

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDiffusionScaling

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: getVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: remeshOut

      double,unsigned int,unsigned int,vector<unsigned int>,vector<double> (*source message field*) Tells the target pool or other entity that the compartment subdivision(meshing) has changed, and that it has to redo its volume and memory allocation accordingly.Arguments are: oldvol, numTotalEntries, startEntry, localIndices, volsThe vols specifies volumes of each local mesh entry. It also specifieshow many meshEntries are present on the local node.The localIndices vector is used for general load balancing only.It has a list of the all meshEntries on current node.If it is empty, we assume block load balancing. In this secondcase the contents of the current node go from startEntry to startEntry + vols.size().

   .. py:attribute:: remeshReacsOut

      void (*source message field*) Tells connected enz or reac that the compartment subdivision(meshing) has changed, and that it has to redo its volume-dependent rate terms like numKf_ accordingly.

   .. py:attribute:: volume

      double (*value field*) Volume of this MeshEntry

   .. py:attribute:: dimensions

      unsigned int (*value field*) number of dimensions of this MeshEntry

   .. py:attribute:: meshType

      unsigned int (*value field*)  The MeshType defines the shape of the mesh entry. 0: Not assigned 1: cuboid 2: cylinder 3. cylindrical shell 4: cylindrical shell segment 5: sphere 6: spherical shell 7: spherical shell segment 8: Tetrahedral

   .. py:attribute:: Coordinates

      vector<double> (*value field*) Coordinates that define current MeshEntry. Depend on MeshType.

   .. py:attribute:: neighbors

      vector<unsigned int> (*value field*) Indices of other MeshEntries that this one connects to

   .. py:attribute:: DiffusionArea

      vector<double> (*value field*) Diffusion area for geometry of interface

   .. py:attribute:: DiffusionScaling

      vector<double> (*value field*) Diffusion scaling for geometry of interface

.. py:class:: MgBlock

   MgBlock: Hodgkin-Huxley type voltage-gated Ion channel. Something like the old tabchannel from GENESIS, but also presents a similar interface as hhchan from GENESIS. 

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on.
 The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: origChannel

       (*destination message field*) 

   .. py:method:: setKMg_A

       (*destination message field*) Assigns field value.

   .. py:method:: getKMg_A

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setKMg_B

       (*destination message field*) Assigns field value.

   .. py:method:: getKMg_B

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCMg

       (*destination message field*) Assigns field value.

   .. py:method:: getCMg

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setIk

       (*destination message field*) Assigns field value.

   .. py:method:: getIk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZk

       (*destination message field*) Assigns field value.

   .. py:method:: getZk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: KMg_A

      double (*value field*) 1/eta

   .. py:attribute:: KMg_B

      double (*value field*) 1/gamma

   .. py:attribute:: CMg

      double (*value field*) [Mg] in mM

   .. py:attribute:: Ik

      double (*value field*) Current through MgBlock

   .. py:attribute:: Zk

      double (*value field*) Charge on ion

.. py:class:: Msg

   .. py:method:: getE1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getE2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSrcFieldsOnE1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDestFieldsOnE2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSrcFieldsOnE2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDestFieldsOnE1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getAdjacent

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: e1

      Id (*value field*) Id of source Element.

   .. py:attribute:: e2

      Id (*value field*) Id of source Element.

   .. py:attribute:: srcFieldsOnE1

      vector<string> (*value field*) Names of SrcFinfos for messages going from e1 to e2. There arematching entries in the destFieldsOnE2 vector

   .. py:attribute:: destFieldsOnE2

      vector<string> (*value field*) Names of DestFinfos for messages going from e1 to e2. There arematching entries in the srcFieldsOnE1 vector

   .. py:attribute:: srcFieldsOnE2

      vector<string> (*value field*) Names of SrcFinfos for messages going from e2 to e1. There arematching entries in the destFieldsOnE1 vector

   .. py:attribute:: destFieldsOnE1

      vector<string> (*value field*) Names of destFinfos for messages going from e2 to e1. There arematching entries in the srcFieldsOnE2 vector

   .. py:attribute:: adjacent

      ObjId,ObjId (*lookup field*) The element adjacent to the specified element

.. py:class:: Mstring

   .. py:method:: setThis

       (*destination message field*) Assigns field value.

   .. py:method:: getThis

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setValue

       (*destination message field*) Assigns field value.

   .. py:method:: getValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: this

      string (*value field*) Access function for entire Mstring object.

   .. py:attribute:: value

      string (*value field*) Access function for value field of Mstring object,which happens also to be the entire contents of the object.

.. py:class:: Nernst

   .. py:method:: getE

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTemperature

       (*destination message field*) Assigns field value.

   .. py:method:: getTemperature

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setValence

       (*destination message field*) Assigns field value.

   .. py:method:: getValence

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCin

       (*destination message field*) Assigns field value.

   .. py:method:: getCin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCout

       (*destination message field*) Assigns field value.

   .. py:method:: getCout

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setScale

       (*destination message field*) Assigns field value.

   .. py:method:: getScale

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: ci

       (*destination message field*) Set internal conc of ion, and immediately send out the updated E

   .. py:method:: co

       (*destination message field*) Set external conc of ion, and immediately send out the updated E

   .. py:attribute:: Eout

      double (*source message field*) Computed reversal potential

   .. py:attribute:: E

      double (*value field*) Computed reversal potential

   .. py:attribute:: Temperature

      double (*value field*) Temperature of cell

   .. py:attribute:: valence

      int (*value field*) Valence of ion in Nernst calculation

   .. py:attribute:: Cin

      double (*value field*) Internal conc of ion

   .. py:attribute:: Cout

      double (*value field*) External conc of ion

   .. py:attribute:: scale

      double (*value field*) Voltage scale factor

.. py:class:: NeuroMesh

   .. py:method:: setCell

       (*destination message field*) Assigns field value.

   .. py:method:: getCell

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSubTree

       (*destination message field*) Assigns field value.

   .. py:method:: getSubTree

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSeparateSpines

       (*destination message field*) Assigns field value.

   .. py:method:: getSeparateSpines

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumSegments

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumDiffCompts

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getParentVoxel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getElecComptList

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getElecComptMap

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getStartVoxelInCompt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getEndVoxelInCompt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDiffLength

       (*destination message field*) Assigns field value.

   .. py:method:: getDiffLength

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGeometryPolicy

       (*destination message field*) Assigns field value.

   .. py:method:: getGeometryPolicy

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: cellPortion

       (*destination message field*) Tells NeuroMesh to mesh up a subpart of a cell. For nowassumed contiguous.The first argument is the cell Id. The second is the wildcardpath of compartments to use for the subpart.

   .. py:attribute:: spineListOut

      Id,vector<Id>,vector<Id>,vector<unsigned int> (*source message field*) Request SpineMesh to construct self based on list of electrical compartments that this NeuroMesh has determined are spine shaft and spine head respectively. Also passes in the info about where each spine is connected to the NeuroMesh. Arguments: Cell Id, shaft compartment Ids, head compartment Ids,index of matching parent voxels for each spine

   .. py:attribute:: psdListOut

      Id,vector<double>,vector<unsigned int> (*source message field*) Tells PsdMesh to build a mesh. Arguments: Cell Id, Coordinates of each psd, index of matching parent voxels for each spineThe coordinates each have 8 entries:xyz of centre of psd, xyz of vector perpendicular to psd, psd diameter,  diffusion distance from parent compartment to PSD

   .. py:attribute:: cell

      Id (*value field*) Id for base element of cell model. Uses this to traverse theentire tree of the cell to build the mesh.

   .. py:attribute:: subTree

      vector<Id> (*value field*) Set of compartments to model. If they happen to be contiguousthen also set up diffusion between the compartments. Can alsohandle cases where the same cell is divided into multiplenon-diffusively-coupled compartments

   .. py:attribute:: separateSpines

      bool (*value field*) Flag: when separateSpines is true, the traversal separates any compartment with the strings 'spine', 'head', 'shaft' or 'neck' in its name,Allows to set up separate mesh for spines, based on the same cell model. Requires for the spineListOut message tobe sent to the target SpineMesh object.

   .. py:attribute:: numSegments

      unsigned int (*value field*) Number of cylindrical/spherical segments in model

   .. py:attribute:: numDiffCompts

      unsigned int (*value field*) Number of diffusive compartments in model

   .. py:attribute:: parentVoxel

      vector<unsigned int> (*value field*) Vector of indices of parents of each voxel.

   .. py:attribute:: elecComptList

      vector<Id> (*value field*) Vector of Ids of all electrical compartments in this NeuroMesh. Ordering is as per the tree structure built in the NeuroMesh, and may differ from Id order. Ordering matches that used for startVoxelInCompt and endVoxelInCompt

   .. py:attribute:: elecComptMap

      vector<Id> (*value field*) Vector of Ids of electrical compartments that map to each voxel. This is necessary because the order of the IDs may differ from the ordering of the voxels. Additionally, there are typically many more voxels than there are electrical compartments. So many voxels point to the same elecCompt.

   .. py:attribute:: startVoxelInCompt

      vector<unsigned int> (*value field*) Index of first voxel that maps to each electrical compartment. Each elecCompt has one or more voxels. The voxels in a compartment are numbered sequentially.

   .. py:attribute:: endVoxelInCompt

      vector<unsigned int> (*value field*) Index of end voxel that maps to each electrical compartment. In keeping with C and Python convention, this is one more than the last voxel. Each elecCompt has one or more voxels. The voxels in a compartment are numbered sequentially.

   .. py:attribute:: diffLength

      double (*value field*) Diffusive length constant to use for subdivisions. The system willattempt to subdivide cell using diffusive compartments ofthe specified diffusion lengths as a maximum.In order to get integral numbersof compartments in each segment, it may subdivide more finely.Uses default of 0.5 microns, that is, half typical lambda.For default, consider a tau of about 1 second for mostreactions, and a diffusion const of about 1e-12 um^2/sec.This gives lambda of 1 micron

   .. py:attribute:: geometryPolicy

      string (*value field*) Policy for how to interpret electrical model geometry (which is a branching 1-dimensional tree) in terms of 3-D constructslike spheres, cylinders, and cones.There are three options, default, trousers, and cylinder:default mode: - Use frustrums of cones. Distal diameter is always from compt dia. - For linear dendrites (no branching), proximal diameter is  diameter of the parent compartment - For branching dendrites and dendrites emerging from soma, proximal diameter is from compt dia. Don't worry about overlap. - Place somatic dendrites on surface of spherical soma, or at ends of cylindrical soma - Place dendritic spines on surface of cylindrical dendrites, not emerging from their middle.trousers mode: - Use frustrums of cones. Distal diameter is always from compt dia. - For linear dendrites (no branching), proximal diameter is  diameter of the parent compartment - For branching dendrites, use a trouser function. Avoid overlap. - For soma, use some variant of trousers. Here we must avoid overlap - For spines, use a way to smoothly merge into parent dend. Radius of curvature should be similar to that of the spine neck. - Place somatic dendrites on surface of spherical soma, or at ends of cylindrical soma - Place dendritic spines on surface of cylindrical dendrites, not emerging from their middle.cylinder mode: - Use cylinders. Diameter is just compartment dia. - Place somatic dendrites on surface of spherical soma, or at ends of cylindrical soma - Place dendritic spines on surface of cylindrical dendrites, not emerging from their middle. - Ignore spatial overlap.

.. py:class:: Neuron

   Neuron - A compartment container

.. py:class:: Neutral

   Neutral: Base class for all MOOSE classes. Providesaccess functions for housekeeping fields and operations, messagetraversal, and so on.

   .. py:method:: parentMsg

       (*destination message field*) Message from Parent Element(s)

   .. py:method:: setThis

       (*destination message field*) Assigns field value.

   .. py:method:: getThis

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setName

       (*destination message field*) Assigns field value.

   .. py:method:: getName

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMe

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getParent

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getChildren

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getPath

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getClassName

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumData

       (*destination message field*) Assigns field value.

   .. py:method:: getNumData

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumField

       (*destination message field*) Assigns field value.

   .. py:method:: getNumField

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getValueFields

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSourceFields

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDestFields

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMsgOut

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMsgIn

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNeighbors

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMsgDests

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMsgDestFunctions

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: childOut

      int (*source message field*) Message to child Elements

   .. py:attribute:: this

      Neutral (*value field*) Access function for entire object

   .. py:attribute:: name

      string (*value field*) Name of object

   .. py:attribute:: me

      ObjId (*value field*) ObjId for current object

   .. py:attribute:: parent

      ObjId (*value field*) Parent ObjId for current object

   .. py:attribute:: children

      vector<Id> (*value field*) vector of ObjIds listing all children of current object

   .. py:attribute:: path

      string (*value field*) text path for object

   .. py:attribute:: className

      string (*value field*) Class Name of object

   .. py:attribute:: numData

      unsigned int (*value field*) # of Data entries on Element.Note that on a FieldElement this does NOT refer to field entries,but to the number of DataEntries on the parent of the FieldElement.

   .. py:attribute:: numField

      unsigned int (*value field*) For a FieldElement: number of entries of self.For a regular Element: One.

   .. py:attribute:: valueFields

      vector<string> (*value field*) List of all value fields on Element.These fields are accessed through the assignment operations in the Python interface.These fields may also be accessed as functions through the set<FieldName> and get<FieldName> commands.

   .. py:attribute:: sourceFields

      vector<string> (*value field*) List of all source fields on Element, that is fields that can act as message sources. 

   .. py:attribute:: destFields

      vector<string> (*value field*) List of all destination fields on Element, that is, fieldsthat are accessed as Element functions.

   .. py:attribute:: msgOut

      vector<ObjId> (*value field*) Messages going out from this Element

   .. py:attribute:: msgIn

      vector<ObjId> (*value field*) Messages coming in to this Element

   .. py:attribute:: neighbors

      string,vector<Id> (*lookup field*) Ids of Elements connected this Element on specified field.

   .. py:attribute:: msgDests

      string,vector<ObjId> (*lookup field*) ObjIds receiving messages from the specified SrcFinfo

   .. py:attribute:: msgDestFunctions

      string,vector<string> (*lookup field*) Matching function names for each ObjId receiving a msg from the specified SrcFinfo

.. py:class:: OneToAllMsg

   .. py:method:: setI1

       (*destination message field*) Assigns field value.

   .. py:method:: getI1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: i1

      unsigned int (*value field*) DataId of source Element.

.. py:class:: OneToOneDataIndexMsg

.. py:class:: OneToOneMsg

.. py:class:: PIDController

   PID feedback controller.PID stands for Proportional-Integral-Derivative. It is used to feedback control dynamical systems. It tries to create a feedback output such that the sensed (measured) parameter is held at command value. Refer to wikipedia (http://wikipedia.org) for details on PID Controller.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: setGain

       (*destination message field*) Assigns field value.

   .. py:method:: getGain

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSaturation

       (*destination message field*) Assigns field value.

   .. py:method:: getSaturation

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCommand

       (*destination message field*) Assigns field value.

   .. py:method:: getCommand

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSensed

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTauI

       (*destination message field*) Assigns field value.

   .. py:method:: getTauI

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTauD

       (*destination message field*) Assigns field value.

   .. py:method:: getTauD

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getError

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIntegral

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getDerivative

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getE_previous

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: commandIn

       (*destination message field*) Command (desired value) input. This is known as setpoint (SP) in control theory.

   .. py:method:: sensedIn

       (*destination message field*) Sensed parameter - this is the one to be tuned. This is known as process variable (PV) in control theory. This comes from the process we are trying to control.

   .. py:method:: gainDest

       (*destination message field*) Destination message to control the PIDController gain dynamically.

   .. py:method:: process

       (*destination message field*) Handle process calls.

   .. py:method:: reinit

       (*destination message field*) Reinitialize the object.

   .. py:attribute:: output

      double (*source message field*) Sends the output of the PIDController. This is known as manipulated variable (MV) in control theory. This should be fed into the process which we are trying to control.

   .. py:attribute:: gain

      double (*value field*) This is the proportional gain (Kp). This tuning parameter scales the proportional term. Larger gain usually results in faster response, but too much will lead to instability and oscillation.

   .. py:attribute:: saturation

      double (*value field*) Bound on the permissible range of output. Defaults to maximum double value.

   .. py:attribute:: command

      double (*value field*) The command (desired) value of the sensed parameter. In control theory this is commonly known as setpoint(SP).

   .. py:attribute:: sensed

      double (*value field*) Sensed (measured) value. This is commonly known as process variable(PV) in control theory.

   .. py:attribute:: tauI

      double (*value field*) The integration time constant, typically = dt. This is actually proportional gain divided by integral gain (Kp/Ki)). Larger Ki (smaller tauI) usually leads to fast elimination of steady state errors at the cost of larger overshoot.

   .. py:attribute:: tauD

      double (*value field*) The differentiation time constant, typically = dt / 4. This is derivative gain (Kd) times proportional gain (Kp). Larger Kd (tauD) decreases overshoot at the cost of slowing down transient response and may lead to instability.

   .. py:attribute:: outputValue

      double (*value field*) Output of the PIDController. This is given by:      gain * ( error + INTEGRAL[ error dt ] / tau_i   + tau_d * d(error)/dt )
Where gain = proportional gain (Kp), tau_i = integral gain (Kp/Ki) and tau_d = derivative gain (Kd/Kp). In control theory this is also known as the manipulated variable (MV)

   .. py:attribute:: error

      double (*value field*) The error term, which is the difference between command and sensed value.

   .. py:attribute:: integral

      double (*value field*) The integral term. It is calculated as INTEGRAL(error dt) = previous_integral + dt * (error + e_previous)/2.

   .. py:attribute:: derivative

      double (*value field*) The derivative term. This is (error - e_previous)/dt.

   .. py:attribute:: e_previous

      double (*value field*) The error term for previous step.

.. py:class:: Pool

   .. py:method:: increment

       (*destination message field*) Increments mol numbers by specified amount. Can be +ve or -ve

   .. py:method:: decrement

       (*destination message field*) Decrements mol numbers by specified amount. Can be +ve or -ve

.. py:class:: PoolBase

   Abstract base class for pools.

   .. py:attribute:: reac

      void (*shared message field*) Connects to reaction

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:attribute:: species

      void (*shared message field*) Shared message for connecting to species objects

   .. py:method:: setN

       (*destination message field*) Assigns field value.

   .. py:method:: getN

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNInit

       (*destination message field*) Assigns field value.

   .. py:method:: getNInit

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDiffConst

       (*destination message field*) Assigns field value.

   .. py:method:: getDiffConst

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMotorConst

       (*destination message field*) Assigns field value.

   .. py:method:: getMotorConst

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setConc

       (*destination message field*) Assigns field value.

   .. py:method:: getConc

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setConcInit

       (*destination message field*) Assigns field value.

   .. py:method:: getConcInit

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setVolume

       (*destination message field*) Assigns field value.

   .. py:method:: getVolume

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSpeciesId

       (*destination message field*) Assigns field value.

   .. py:method:: getSpeciesId

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: reacDest

       (*destination message field*) Handles reaction input

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: handleMolWt

       (*destination message field*) Separate finfo to assign molWt, and consequently diffusion const.Should only be used in SharedMsg with species.

   .. py:attribute:: nOut

      double (*source message field*) Sends out # of molecules in pool on each timestep

   .. py:attribute:: requestMolWt

      void (*source message field*) Requests Species object for mol wt

   .. py:attribute:: n

      double (*value field*) Number of molecules in pool

   .. py:attribute:: nInit

      double (*value field*) Initial value of number of molecules in pool

   .. py:attribute:: diffConst

      double (*value field*) Diffusion constant of molecule

   .. py:attribute:: motorConst

      double (*value field*) Motor transport rate molecule. + is away from soma, - is towards soma. Only relevant for ZombiePool subclasses.

   .. py:attribute:: conc

      double (*value field*) Concentration of molecules in this pool

   .. py:attribute:: concInit

      double (*value field*) Initial value of molecular concentration in pool

   .. py:attribute:: volume

      double (*value field*) Volume of compartment. Units are SI. Utility field, the actual volume info is stored on a volume mesh entry in the parent compartment.This mapping is implicit: the parent compartment must be somewhere up the element tree, and must have matching mesh entries. If the compartment isn'tavailable the volume is just taken as 1

   .. py:attribute:: speciesId

      unsigned int (*value field*) Species identifier for this mol pool. Eventually link to ontology.

.. py:class:: PostMaster

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: getNumNodes

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMyNode

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBufferSize

       (*destination message field*) Assigns field value.

   .. py:method:: getBufferSize

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: numNodes

      unsigned int (*value field*) Returns number of nodes that simulation runs on.

   .. py:attribute:: myNode

      unsigned int (*value field*) Returns index of current node.

   .. py:attribute:: bufferSize

      unsigned int (*value field*) Size of the send a receive buffers for each node.

.. py:class:: PsdMesh

   .. py:method:: setThickness

       (*destination message field*) Assigns field value.

   .. py:method:: getThickness

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: psdList

       (*destination message field*) Specifies the geometry of the spine,and the associated parent voxelArguments: cell container, disk params vector with 8 entriesper psd, parent voxel index 

   .. py:attribute:: thickness

      double (*value field*) An assumed thickness for PSD. The volume is computed as thePSD area passed in to each PSD, times this value.defaults to 50 nanometres. For reference, membranes are 5 nm.

.. py:class:: PulseGen

   PulseGen: general purpose pulse generator. This can generate any number of pulses with specified level and duration.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBaseLevel

       (*destination message field*) Assigns field value.

   .. py:method:: getBaseLevel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFirstLevel

       (*destination message field*) Assigns field value.

   .. py:method:: getFirstLevel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFirstWidth

       (*destination message field*) Assigns field value.

   .. py:method:: getFirstWidth

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFirstDelay

       (*destination message field*) Assigns field value.

   .. py:method:: getFirstDelay

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSecondLevel

       (*destination message field*) Assigns field value.

   .. py:method:: getSecondLevel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSecondWidth

       (*destination message field*) Assigns field value.

   .. py:method:: getSecondWidth

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSecondDelay

       (*destination message field*) Assigns field value.

   .. py:method:: getSecondDelay

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCount

       (*destination message field*) Assigns field value.

   .. py:method:: getCount

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTrigMode

       (*destination message field*) Assigns field value.

   .. py:method:: getTrigMode

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLevel

       (*destination message field*) Assigns field value.

   .. py:method:: getLevel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setWidth

       (*destination message field*) Assigns field value.

   .. py:method:: getWidth

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDelay

       (*destination message field*) Assigns field value.

   .. py:method:: getDelay

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: input

       (*destination message field*) Handle incoming input that determines gating/triggering onset.

   .. py:method:: levelIn

       (*destination message field*) Handle level value coming from other objects

   .. py:method:: widthIn

       (*destination message field*) Handle width value coming from other objects

   .. py:method:: delayIn

       (*destination message field*) Handle delay value coming from other objects

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: output

      double (*source message field*) Current output level.

   .. py:attribute:: outputValue

      double (*value field*) Output amplitude

   .. py:attribute:: baseLevel

      double (*value field*) Basal level of the stimulus

   .. py:attribute:: firstLevel

      double (*value field*) Amplitude of the first pulse in a sequence

   .. py:attribute:: firstWidth

      double (*value field*) Width of the first pulse in a sequence

   .. py:attribute:: firstDelay

      double (*value field*) Delay to start of the first pulse in a sequence

   .. py:attribute:: secondLevel

      double (*value field*) Amplitude of the second pulse in a sequence

   .. py:attribute:: secondWidth

      double (*value field*) Width of the second pulse in a sequence

   .. py:attribute:: secondDelay

      double (*value field*) Delay to start of of the second pulse in a sequence

   .. py:attribute:: count

      unsigned int (*value field*) Number of pulses in a sequence

   .. py:attribute:: trigMode

      unsigned int (*value field*) Trigger mode for pulses in the sequence.
 0 : free-running mode where it keeps looping its output
 1 : external trigger, where it is triggered by an external input (and stops after creating the first train of pulses)
 2 : external gate mode, where it keeps generating the pulses in a loop as long as the input is high.

   .. py:attribute:: level

      unsigned int,double (*lookup field*) Level of the pulse at specified index

   .. py:attribute:: width

      unsigned int,double (*lookup field*) Width of the pulse at specified index

   .. py:attribute:: delay

      unsigned int,double (*lookup field*) Delay of the pulse at specified index

.. py:class:: RC

   RC circuit: a series resistance R shunted by a capacitance C.

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process messages from the scheduler objects.The first entry in the shared msg is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt and so on. The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo. 

   .. py:method:: setV0

       (*destination message field*) Assigns field value.

   .. py:method:: getV0

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setR

       (*destination message field*) Assigns field value.

   .. py:method:: getR

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setC

       (*destination message field*) Assigns field value.

   .. py:method:: getC

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInject

       (*destination message field*) Assigns field value.

   .. py:method:: getInject

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: injectIn

       (*destination message field*) Receives input to the RC circuit. All incoming messages are summed up to give the total input current.

   .. py:method:: process

       (*destination message field*) Handles process call.

   .. py:method:: reinit

       (*destination message field*) Handle reinitialization

   .. py:attribute:: output

      double (*source message field*) Current output level.

   .. py:attribute:: V0

      double (*value field*) Initial value of 'state'

   .. py:attribute:: R

      double (*value field*) Series resistance of the RC circuit.

   .. py:attribute:: C

      double (*value field*) Parallel capacitance of the RC circuit.

   .. py:attribute:: state

      double (*value field*) Output value of the RC circuit. This is the voltage across the capacitor.

   .. py:attribute:: inject

      double (*value field*) Input value to the RC circuit.This is handled as an input current to the circuit.

.. py:class:: Reac

.. py:class:: ReacBase

   Base class for reactions. Provides the MOOSE APIfunctions, but ruthlessly refers almost all of them to derivedclasses, which have to provide the man page output.

   .. py:attribute:: sub

      void (*shared message field*) Connects to substrate pool

   .. py:attribute:: prd

      void (*shared message field*) Connects to substrate pool

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setNumKf

       (*destination message field*) Assigns field value.

   .. py:method:: getNumKf

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumKb

       (*destination message field*) Assigns field value.

   .. py:method:: getNumKb

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setKf

       (*destination message field*) Assigns field value.

   .. py:method:: getKf

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setKb

       (*destination message field*) Assigns field value.

   .. py:method:: getKb

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumSubstrates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumProducts

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: subDest

       (*destination message field*) Handles # of molecules of substrate

   .. py:method:: prdDest

       (*destination message field*) Handles # of molecules of product

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: subOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: prdOut

      double,double (*source message field*) Sends out increment of molecules on product each timestep

   .. py:attribute:: numKf

      double (*value field*) Forward rate constant, in # units

   .. py:attribute:: numKb

      double (*value field*) Reverse rate constant, in # units

   .. py:attribute:: Kf

      double (*value field*) Forward rate constant, in concentration units

   .. py:attribute:: Kb

      double (*value field*) Reverse rate constant, in concentration units

   .. py:attribute:: numSubstrates

      unsigned int (*value field*) Number of substrates of reaction

   .. py:attribute:: numProducts

      unsigned int (*value field*) Number of products of reaction

.. py:class:: Shell

   .. py:method:: setclock

       (*destination message field*) Assigns clock ticks. Args: tick#, dt

   .. py:method:: create

       (*destination message field*) create( class, parent, newElm, name, numData, isGlobal )

   .. py:method:: delete

       (*destination message field*) Destroys Element, all its messages, and all its children. Args: Id

   .. py:method:: copy

       (*destination message field*) handleCopy( vector< Id > args, string newName, unsigned int nCopies, bool toGlobal, bool copyExtMsgs ):  The vector< Id > has Id orig, Id newParent, Id newElm. This function copies an Element and all its children to a new parent. May also expand out the original into nCopies copies. Normally all messages within the copy tree are also copied.  If the flag copyExtMsgs is true, then all msgs going out are also copied.

   .. py:method:: move

       (*destination message field*) handleMove( Id orig, Id newParent ): moves an Element to a new parent

   .. py:method:: addMsg

       (*destination message field*) Makes a msg. Arguments are: msgtype, src object, src field, dest object, dest field

   .. py:method:: quit

       (*destination message field*) Stops simulation running and quits the simulator

   .. py:method:: useClock

       (*destination message field*) Deals with assignment of path to a given clock. Arguments: path, field, tick number. 

.. py:class:: SingleMsg

   .. py:method:: setI1

       (*destination message field*) Assigns field value.

   .. py:method:: getI1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setI2

       (*destination message field*) Assigns field value.

   .. py:method:: getI2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: i1

      unsigned int (*value field*) Index of source object.

   .. py:attribute:: i2

      unsigned int (*value field*) Index of dest object.

.. py:class:: SparseMsg

   .. py:method:: getNumRows

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumColumns

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumEntries

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setProbability

       (*destination message field*) Assigns field value.

   .. py:method:: getProbability

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setSeed

       (*destination message field*) Assigns field value.

   .. py:method:: getSeed

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRandomConnectivity

       (*destination message field*) Assigns connectivity with specified probability and seed

   .. py:method:: setEntry

       (*destination message field*) Assigns single row,column value

   .. py:method:: unsetEntry

       (*destination message field*) Clears single row,column entry

   .. py:method:: clear

       (*destination message field*) Clears out the entire matrix

   .. py:method:: transpose

       (*destination message field*) Transposes the sparse matrix

   .. py:method:: pairFill

       (*destination message field*) Fills entire matrix using pairs of (x,y) indices to indicate presence of a connection. If the target is a FieldElement itautomagically assigns FieldIndices.

   .. py:method:: tripletFill

       (*destination message field*) Fills entire matrix using triplets of (x,y,fieldIndex) to fully specify every connection in the sparse matrix.

   .. py:attribute:: numRows

      unsigned int (*value field*) Number of rows in matrix.

   .. py:attribute:: numColumns

      unsigned int (*value field*) Number of columns in matrix.

   .. py:attribute:: numEntries

      unsigned int (*value field*) Number of Entries in matrix.

   .. py:attribute:: probability

      double (*value field*) connection probability for random connectivity.

   .. py:attribute:: seed

      long (*value field*) Random number seed for generating probabilistic connectivity.

.. py:class:: Species

   .. py:attribute:: pool

      void (*shared message field*) Connects to pools of this Species type

   .. py:method:: setMolWt

       (*destination message field*) Assigns field value.

   .. py:method:: getMolWt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: handleMolWtRequest

       (*destination message field*) Handle requests for molWt.

   .. py:attribute:: molWtOut

      double (*source message field*) returns molWt.

   .. py:attribute:: molWt

      double (*value field*) Molecular weight of species

.. py:class:: SpikeGen

   SpikeGen object, for detecting threshold crossings.The threshold detection can work in multiple modes.

    If the refractT < 0.0, then it fires an event only at the rising edge of the input voltage waveform

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process message from scheduler

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: Vm

       (*destination message field*) Handles Vm message coming in from compartment

   .. py:method:: setThreshold

       (*destination message field*) Assigns field value.

   .. py:method:: getThreshold

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setRefractT

       (*destination message field*) Assigns field value.

   .. py:method:: getRefractT

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setAbs_refract

       (*destination message field*) Assigns field value.

   .. py:method:: getAbs_refract

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getHasFired

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEdgeTriggered

       (*destination message field*) Assigns field value.

   .. py:method:: getEdgeTriggered

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: spikeOut

      double (*source message field*) Sends out a trigger for an event.

   .. py:attribute:: threshold

      double (*value field*) Spiking threshold, must cross it going up

   .. py:attribute:: refractT

      double (*value field*) Refractory Time.

   .. py:attribute:: abs_refract

      double (*value field*) Absolute refractory time. Synonym for refractT.

   .. py:attribute:: hasFired

      bool (*value field*) True if SpikeGen has just fired

   .. py:attribute:: edgeTriggered

      bool (*value field*) When edgeTriggered = 0, the SpikeGen will fire an event in each timestep while incoming Vm is > threshold and at least abs_refracttime has passed since last event. This may be problematic if the incoming Vm remains above threshold for longer than abs_refract. Setting edgeTriggered to 1 resolves this as the SpikeGen generatesan event only on the rising edge of the incoming Vm and will remain idle unless the incoming Vm goes below threshold.

.. py:class:: SpineMesh

   .. py:method:: getParentVoxel

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: spineList

       (*destination message field*) Specifies the list of electrical compartments for the spine,and the associated parent voxelArguments: cell container, shaft compartments, head compartments, parent voxel index 

   .. py:attribute:: parentVoxel

      vector<unsigned int> (*value field*) Vector of indices of proximal voxels within this mesh.Spines are at present modeled with just one compartment,so each entry in this vector is always set to EMPTY == -1U

.. py:class:: Stats

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: getMean

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSdev

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSum

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNum

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: mean

      double (*value field*) Mean of all sampled values.

   .. py:attribute:: sdev

      double (*value field*) Standard Deviation of all sampled values.

   .. py:attribute:: sum

      double (*value field*) Sum of all sampled values.

   .. py:attribute:: num

      unsigned int (*value field*) Number of all sampled values.

.. py:class:: SteadyState

   SteadyState: works out a steady-state value for a reaction system. It uses GSL heavily, and isn't even compiled if the flag isn't set. It finds the ss value closest to the initial conditions, defined by current molecular concentrations.If you want to find multiple stable states, use the MultiStable object,which operates a SteadyState object to find multiple states.If you want to carry out a dose-response calculation, use the DoseResponse object.If you want to follow a stable state in phase space, use the StateTrajectory object. 

   .. py:method:: setStoich

       (*destination message field*) Assigns field value.

   .. py:method:: getStoich

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getBadStoichiometry

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIsInitialized

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNIter

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getStatus

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMaxIter

       (*destination message field*) Assigns field value.

   .. py:method:: getMaxIter

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setConvergenceCriterion

       (*destination message field*) Assigns field value.

   .. py:method:: getConvergenceCriterion

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumVarPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getRank

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getStateType

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNNegEigenvalues

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNPosEigenvalues

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSolutionStatus

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTotal

       (*destination message field*) Assigns field value.

   .. py:method:: getTotal

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getEigenvalues

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setupMatrix

       (*destination message field*) This function initializes and rebuilds the matrices used in the calculation.

   .. py:method:: settle

       (*destination message field*) Finds the nearest steady state to the current initial conditions. This function rebuilds the entire calculation only if the object has not yet been initialized.

   .. py:method:: resettle

       (*destination message field*) Finds the nearest steady state to the current initial conditions. This function rebuilds the entire calculation 

   .. py:method:: showMatrices

       (*destination message field*) Utility function to show the matrices derived for the calculations on the reaction system. Shows the Nr, gamma, and total matrices

   .. py:method:: randomInit

       (*destination message field*) Generate random initial conditions consistent with the massconservation rules. Typically invoked in order to scanstates

   .. py:attribute:: stoich

      Id (*value field*) Specify the Id of the stoichiometry system to use

   .. py:attribute:: badStoichiometry

      bool (*value field*) Bool: True if there is a problem with the stoichiometry

   .. py:attribute:: isInitialized

      bool (*value field*) True if the model has been initialized successfully

   .. py:attribute:: nIter

      unsigned int (*value field*) Number of iterations done by steady state solver

   .. py:attribute:: status

      string (*value field*) Status of solver

   .. py:attribute:: maxIter

      unsigned int (*value field*) Max permissible number of iterations to try before giving up

   .. py:attribute:: convergenceCriterion

      double (*value field*) Fractional accuracy required to accept convergence

   .. py:attribute:: numVarPools

      unsigned int (*value field*) Number of variable molecules in reaction system.

   .. py:attribute:: rank

      unsigned int (*value field*) Number of independent molecules in reaction system

   .. py:attribute:: stateType

      unsigned int (*value field*) 0: stable; 1: unstable; 2: saddle; 3: osc?; 4: one near-zero eigenvalue; 5: other

   .. py:attribute:: nNegEigenvalues

      unsigned int (*value field*) Number of negative eigenvalues: indicates type of solution

   .. py:attribute:: nPosEigenvalues

      unsigned int (*value field*) Number of positive eigenvalues: indicates type of solution

   .. py:attribute:: solutionStatus

      unsigned int (*value field*) 0: Good; 1: Failed to find steady states; 2: Failed to find eigenvalues

   .. py:attribute:: total

      unsigned int,double (*lookup field*) Totals table for conservation laws. The exact mapping ofthis to various sums of molecules is given by the conservation matrix, and is currently a bit opaque.The value of 'total' is set to initial conditions whenthe 'SteadyState::settle' function is called.Assigning values to the total is a special operation:it rescales the concentrations of all the affectedmolecules so that they are at the specified total.This happens the next time 'settle' is called.

   .. py:attribute:: eigenvalues

      unsigned int,double (*lookup field*) Eigenvalues computed for steady state

.. py:class:: StimulusTable

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setStartTime

       (*destination message field*) Assigns field value.

   .. py:method:: getStartTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setStopTime

       (*destination message field*) Assigns field value.

   .. py:method:: getStopTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setLoopTime

       (*destination message field*) Assigns field value.

   .. py:method:: getLoopTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setStepSize

       (*destination message field*) Assigns field value.

   .. py:method:: getStepSize

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setStepPosition

       (*destination message field*) Assigns field value.

   .. py:method:: getStepPosition

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDoLoop

       (*destination message field*) Assigns field value.

   .. py:method:: getDoLoop

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: output

      double (*source message field*) Sends out tabulated data according to lookup parameters.

   .. py:attribute:: startTime

      double (*value field*) Start time used when table is emitting values. For lookupvalues below this, the table just sends out its zero entry.Corresponds to zeroth entry of table.

   .. py:attribute:: stopTime

      double (*value field*) Time to stop emitting values.If time exceeds this, then the table sends out its last entry.The stopTime corresponds to the last entry of table.

   .. py:attribute:: loopTime

      double (*value field*) If looping, this is the time between successive cycle starts.Defaults to the difference between stopTime and startTime, so that the output waveform cycles with precisely the same duration as the table contents.If larger than stopTime - startTime, then it pauses at the last table value till it is time to go around again.If smaller than stopTime - startTime, then it begins the next cycle even before the first one has reached the end of the table.

   .. py:attribute:: stepSize

      double (*value field*) Increment in lookup (x) value on every timestep. If it isless than or equal to zero, the StimulusTable uses the current timeas the lookup value.

   .. py:attribute:: stepPosition

      double (*value field*) Current value of lookup (x) value.If stepSize is less than or equal to zero, this is set tothe current time to use as the lookup value.

   .. py:attribute:: doLoop

      bool (*value field*) Flag: Should it loop around to startTime once it has reachedstopTime. Default (zero) is to do a single pass.

.. py:class:: Stoich

   .. py:method:: setPath

       (*destination message field*) Assigns field value.

   .. py:method:: getPath

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setKsolve

       (*destination message field*) Assigns field value.

   .. py:method:: getKsolve

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDsolve

       (*destination message field*) Assigns field value.

   .. py:method:: getDsolve

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCompartment

       (*destination message field*) Assigns field value.

   .. py:method:: getCompartment

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getEstimatedDt

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumVarPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumAllPools

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getPoolIdMap

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getNumRates

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getMatrixEntry

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getColumnIndex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getRowStart

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: unzombify

       (*destination message field*) Restore all zombies to their native state

   .. py:attribute:: path

      string (*value field*) Wildcard path for reaction system handled by Stoich

   .. py:attribute:: ksolve

      Id (*value field*) Id of Kinetic reaction solver class that works with this Stoich.  Must be of class Ksolve, or Gsolve (at present)  Must be assigned before the path is set.

   .. py:attribute:: dsolve

      Id (*value field*) Id of Diffusion solver class that works with this Stoich. Must be of class Dsolve  If left unset then the system will be assumed to work in a non-diffusive, well-stirred cell. If it is going to be  used it must be assigned before the path is set.

   .. py:attribute:: compartment

      Id (*value field*) Id of chemical compartment class that works with this Stoich. Must be derived from class ChemCompt. If left unset then the system will be assumed to work in a non-diffusive, well-stirred cell. If it is going to be  used it must be assigned before the path is set.

   .. py:attribute:: estimatedDt

      double (*value field*) Estimated timestep for reac system based on Euler error

   .. py:attribute:: numVarPools

      unsigned int (*value field*) Number of time-varying pools to be computed by the numerical engine

   .. py:attribute:: numAllPools

      unsigned int (*value field*) Total number of pools handled by the numerical engine. This includes variable ones, buffered ones, and functions

   .. py:attribute:: poolIdMap

      vector<unsigned int> (*value field*) Map to look up the index of the pool from its Id.poolIndex = poolIdMap[ Id::value() - poolOffset ] where the poolOffset is the smallest Id::value. poolOffset is passed back as the last entry of this vector. Any Ids that are not pools return EMPTY=~0. 

   .. py:attribute:: numRates

      unsigned int (*value field*) Total number of rate terms in the reaction system.

   .. py:attribute:: matrixEntry

      vector<int> (*value field*) The non-zero matrix entries in the sparse matrix. Theircolumn indices are in a separate vector and the rowinformatino in a third

   .. py:attribute:: columnIndex

      vector<unsigned int> (*value field*) Column Index of each matrix entry

   .. py:attribute:: rowStart

      vector<unsigned int> (*value field*) Row start for each block of entries and column indices

.. py:class:: SumFunc

   SumFunc object. Adds up all inputs

.. py:class:: SymCompartment

   SymCompartment object, for branching neuron models. In symmetric

   compartments the axial resistance is equally divided on two sides of

   

    you must use a fixed-width font like Courier for correct rendition of the diagrams below.]

                                          

            Ra/2    B    Ra/2               

          A-/\/\/\_____/\/\/\-- C           

                    |                      

                ____|____                  

               |         |                 

               |         \                 

               |         / Rm              

              ---- Cm    \                 

              ----       /                 

               |         |                 

               |       _____               

               |        ---  Em            

               |_________|                 

                   |                       

                 __|__                     

                 /////                     

                                          

                                          

   In case of branching, the B-C part of the parent's axial resistance

   forms a Y with the A-B part of the children.

                                  B'              

                                  |               

                                  /               

                                  \              

                                  /               

                                  \              

                                  /               

                                  |A'             

                   B              |               

     A-----/\/\/\-----/\/\/\------|C        

                                  |               

                                  |A"            

                                  /               

                                  \              

                                  /               

                                  \              

                                  /               

                                  |               

                                  B"             

   As per basic circuit analysis techniques, the C node is replaced using

   star-mesh transform. This requires all sibling compartments at a

   branch point to be connected via 'sibling' messages by the user (or

   by the cell reader in case of prototypes). For the same reason, the

   child compartment must be connected to the parent by

   distal-proximal message pair. The calculation of the

   coefficient for computing equivalent resistances in the mesh is done

   at reinit.

   .. py:attribute:: proximal

      void (*shared message field*) This is a shared message between symmetric compartments.
It goes from the proximal end of the current compartment to
distal end of the compartment closer to the soma.


   .. py:attribute:: distal

      void (*shared message field*) This is a shared message between symmetric compartments.
It goes from the distal end of the current compartment to the 
 proximal end of one further from the soma. 
The Ra values collected from children and
sibling nodes are used for computing the equivalent resistance 
between each pair of nodes using star-mesh transformation.
Mathematically this is the same as the proximal message, but
the distinction is important for traversal and clarity.


   .. py:attribute:: sibling

      void (*shared message field*) This is a shared message between symmetric compartments.
Conceptually, this goes from the proximal end of the current 
compartment to the proximal end of a sibling compartment 
on a branch in a dendrite. However,
this works out to the same as a 'distal' message in terms of 
equivalent circuit.  The Ra values collected from siblings 
and parent node are used for 
computing the equivalent resistance between each pair of
nodes using star-mesh transformation.


   .. py:attribute:: sphere

      void (*shared message field*) This is a shared message between a spherical compartment 
(typically soma) and a number of evenly spaced cylindrical 
compartments, typically primary dendrites.
The sphere contributes the usual Ra/2 to the resistance
between itself and children. The child compartments 
do not connect across to each other
through sibling messages. Instead they just connect to the soma
through the 'proximalOnly' message


   .. py:attribute:: cylinder

      void (*shared message field*) This is a shared message between a cylindrical compartment 
(typically a dendrite) and a number of evenly spaced child 
compartments, typically dendritic spines, protruding from the
curved surface of the cylinder. We assume that the resistance
from the cylinder curved surface to its axis is negligible.
The child compartments do not need to connect across to each 
other through sibling messages. Instead they just connect to the
parent dendrite through the 'proximalOnly' message


   .. py:attribute:: proximalOnly

      void (*shared message field*) This is a shared message between a dendrite and a parent
compartment whose offspring are spatially separated from each
other. For example, evenly spaced dendrites emerging from a soma
or spines emerging from a common parent dendrite. In these cases
the sibling dendrites do not need to connect to each other
through 'sibling' messages. Instead they just connect to the
parent compartment (soma or dendrite) through this message


   .. py:method:: raxialSym

       (*destination message field*) Expects Ra and Vm from other compartment.

   .. py:method:: sumRaxial

       (*destination message field*) Expects Ra from other compartment.

   .. py:method:: raxialSym

       (*destination message field*) Expects Ra and Vm from other compartment.

   .. py:method:: sumRaxial

       (*destination message field*) Expects Ra from other compartment.

   .. py:method:: raxialSym

       (*destination message field*) Expects Ra and Vm from other compartment.

   .. py:method:: sumRaxial

       (*destination message field*) Expects Ra from other compartment.

   .. py:method:: raxialSphere

       (*destination message field*) Expects Ra and Vm from other compartment. This is a special case when
other compartments are evenly distributed on a spherical compartment.

   .. py:method:: raxialCylinder

       (*destination message field*) Expects Ra and Vm from other compartment. This is a special case when
other compartments are evenly distributed on the curved surface of the cylindrical compartment, so we assume that the cylinder does not add any further resistance.

   .. py:method:: raxialSphere

       (*destination message field*) Expects Ra and Vm from other compartment. This is a special case when
other compartments are evenly distributed on a spherical compartment.

   .. py:attribute:: proximalOut

      double,double (*source message field*) Sends out Ra and Vm on each timestep, on the proximalend of a compartment. That is, this end should be pointed toward the soma. Mathematically the same as raxialOutbut provides a logical orientation of the dendrite.One can traverse proximalOut messages to get to the soma.

   .. py:attribute:: sumRaxialOut

      double (*source message field*) Sends out Ra

   .. py:attribute:: distalOut

      double,double (*source message field*) Sends out Ra and Vm on each timestep, on the distal endof a compartment. This end should be pointed away from thesoma. Mathematically the same as proximalOut, but givesan orientation to the dendrite and helps traversal.

   .. py:attribute:: sumRaxialOut

      double (*source message field*) Sends out Ra

   .. py:attribute:: distalOut

      double,double (*source message field*) Sends out Ra and Vm on each timestep, on the distal endof a compartment. This end should be pointed away from thesoma. Mathematically the same as proximalOut, but givesan orientation to the dendrite and helps traversal.

   .. py:attribute:: sumRaxialOut

      double (*source message field*) Sends out Ra

   .. py:attribute:: distalOut

      double,double (*source message field*) Sends out Ra and Vm on each timestep, on the distal endof a compartment. This end should be pointed away from thesoma. Mathematically the same as proximalOut, but givesan orientation to the dendrite and helps traversal.

   .. py:attribute:: cylinderOut

      double,double (*source message field*) Sends out Ra and Vm to compartments (typically spines) on thecurved surface of a cylinder. Ra is set to nearly zero,since we assume that the resistance from axis to surface isnegligible.

   .. py:attribute:: proximalOut

      double,double (*source message field*) Sends out Ra and Vm on each timestep, on the proximalend of a compartment. That is, this end should be pointed toward the soma. Mathematically the same as raxialOutbut provides a logical orientation of the dendrite.One can traverse proximalOut messages to get to the soma.

.. py:class:: SynChan

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process message from scheduler

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setTau1

       (*destination message field*) Assigns field value.

   .. py:method:: getTau1

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau2

       (*destination message field*) Assigns field value.

   .. py:method:: getTau2

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNormalizeWeights

       (*destination message field*) Assigns field value.

   .. py:method:: getNormalizeWeights

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: activation

       (*destination message field*) Sometimes we want to continuously activate the channel

   .. py:method:: modulator

       (*destination message field*) Modulate channel response

   .. py:attribute:: tau1

      double (*value field*) Decay time constant for the synaptic conductance, tau1 >= tau2.

   .. py:attribute:: tau2

      double (*value field*) Rise time constant for the synaptic conductance, tau1 >= tau2.

   .. py:attribute:: normalizeWeights

      bool (*value field*) Flag. If true, the overall conductance is normalized by the number of individual synapses in this SynChan object.

.. py:class:: SynChanBase

   SynChanBase: Base class for assorted ion channels.Presents a common interface for all of them. 

   .. py:attribute:: channel

      void (*shared message field*) This is a shared message to couple channel to compartment. The first entry is a MsgSrc to send Gk and Ek to the compartment The second entry is a MsgDest for Vm from the compartment.

   .. py:attribute:: ghk

      void (*shared message field*) Message to Goldman-Hodgkin-Katz object

   .. py:method:: Vm

       (*destination message field*) Handles Vm message coming in from compartment

   .. py:method:: Vm

       (*destination message field*) Handles Vm message coming in from compartment

   .. py:method:: setGbar

       (*destination message field*) Assigns field value.

   .. py:method:: getGbar

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEk

       (*destination message field*) Assigns field value.

   .. py:method:: getEk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGk

       (*destination message field*) Assigns field value.

   .. py:method:: getGk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setBufferTime

       (*destination message field*) Assigns field value.

   .. py:method:: getBufferTime

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: channelOut

      double,double (*source message field*) Sends channel variables Gk and Ek to compartment

   .. py:attribute:: permeabilityOut

      double (*source message field*) Conductance term going out to GHK object

   .. py:attribute:: IkOut

      double (*source message field*) Channel current. This message typically goes to concenobjects that keep track of ion concentration.

   .. py:attribute:: Gbar

      double (*value field*) Maximal channel conductance

   .. py:attribute:: Ek

      double (*value field*) Reversal potential of channel

   .. py:attribute:: Gk

      double (*value field*) Channel conductance variable

   .. py:attribute:: Ik

      double (*value field*) Channel current variable

   .. py:attribute:: bufferTime

      double (*value field*) Duration of spike buffer.

.. py:class:: SynHandler

   .. py:method:: setNumSynapses

       (*destination message field*) Assigns field value.

   .. py:method:: getNumSynapses

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumSynapse

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumSynapse

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:attribute:: numSynapses

      unsigned int (*value field*) Number of synapses on SynHandler. Duplicate field for num_synapse

.. py:class:: Synapse

   Synapse using ring buffer for events.

   .. py:method:: setWeight

       (*destination message field*) Assigns field value.

   .. py:method:: getWeight

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setDelay

       (*destination message field*) Assigns field value.

   .. py:method:: getDelay

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: addSpike

       (*destination message field*) Handles arriving spike messages, inserts into event queue.

   .. py:attribute:: weight

      double (*value field*) Synaptic weight

   .. py:attribute:: delay

      double (*value field*) Axonal propagation delay to this synapse

.. py:class:: Table

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setThreshold

       (*destination message field*) Assigns field value.

   .. py:method:: getThreshold

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: input

       (*destination message field*) Fills data into table. Also handles data sent back following request

   .. py:method:: spike

       (*destination message field*) Fills spike timings into the Table. Signal has to exceed thresh

   .. py:method:: process

       (*destination message field*) Handles process call, updates internal time stamp.

   .. py:method:: reinit

       (*destination message field*) Handles reinit call.

   .. py:attribute:: requestOut

      Pd (*source message field*) Sends request for a field to target object

   .. py:attribute:: threshold

      double (*value field*) threshold used when Table acts as a buffer for spikes

.. py:class:: TableBase

   .. py:method:: setVector

       (*destination message field*) Assigns field value.

   .. py:method:: getVector

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getOutputValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSize

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: linearTransform

       (*destination message field*) Linearly scales and offsets data. Scale first, then offset.

   .. py:method:: xplot

       (*destination message field*) Dumps table contents to xplot-format file. Argument 1 is filename, argument 2 is plotname

   .. py:method:: plainPlot

       (*destination message field*) Dumps table contents to single-column ascii file. Uses scientific notation. Argument 1 is filename

   .. py:method:: loadCSV

       (*destination message field*) Reads a single column from a CSV file. Arguments: filename, column#, starting row#, separator

   .. py:method:: loadXplot

       (*destination message field*) Reads a single plot from an xplot file. Arguments: filename, plotnameWhen the file has 2 columns, the 2nd column is loaded.

   .. py:method:: loadXplotRange

       (*destination message field*) Reads a single plot from an xplot file, and selects a subset of points from it. Arguments: filename, plotname, startindex, endindexUses C convention: startindex included, endindex not included.When the file has 2 columns, the 2nd column is loaded.

   .. py:method:: compareXplot

       (*destination message field*) Reads a plot from an xplot file and compares with contents of TableBase.Result is put in 'output' field of table.If the comparison fails (e.g., due to zero entries), the return value is -1.Arguments: filename, plotname, comparison_operationOperations: rmsd (for RMSDifference), rmsr (RMSratio ), dotp (Dot product, not yet implemented).

   .. py:method:: compareVec

       (*destination message field*) Compares contents of TableBase with a vector of doubles.Result is put in 'output' field of table.If the comparison fails (e.g., due to zero entries), the return value is -1.Arguments: Other vector, comparison_operationOperations: rmsd (for RMSDifference), rmsr (RMSratio ), dotp (Dot product, not yet implemented).

   .. py:method:: clearVec

       (*destination message field*) Handles request to clear the data vector

   .. py:attribute:: vector

      vector<double> (*value field*) vector with all table entries

   .. py:attribute:: outputValue

      double (*value field*) Output value holding current table entry or output of a calculation

   .. py:attribute:: size

      unsigned int (*value field*) size of table. Note that this is the number of x divisions +1since it must represent the largest value as well as thesmallest

   .. py:attribute:: y

      unsigned int,double (*lookup field*) Value of table at specified index

.. py:class:: TimeTable

   TimeTable: Read in spike times from file and send out eventOut messages

   at the specified times.

   .. py:attribute:: proc

      void (*shared message field*) Shared message for process and reinit

   .. py:method:: setFilename

       (*destination message field*) Assigns field value.

   .. py:method:: getFilename

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMethod

       (*destination message field*) Assigns field value.

   .. py:method:: getMethod

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getState

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: process

       (*destination message field*) Handle process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:attribute:: eventOut

      double (*source message field*) Sends out spike time if it falls in current timestep.

   .. py:attribute:: filename

      string (*value field*) File to read lookup data from. The file should be contain two columns
separated by any space character.

   .. py:attribute:: method

      int (*value field*) Method to use for filling up the entries. Currently only method 4
(loading from file) is supported.

   .. py:attribute:: state

      double (*value field*) Current state of the time table.

.. py:class:: Unsigned

   Variable for storing values.

   .. py:method:: setValue

       (*destination message field*) Assigns field value.

   .. py:method:: getValue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: value

      unsigned long (*value field*) Variable value

.. py:class:: VClamp

   Voltage clamp object for holding neuronal compartments at a specific voltage. This implementation uses a builtin RC circuit to filter the

   command input and then use a PID to bring the sensed voltage (Vm from

   compartment) to the filtered command potential.

    Connect the `currentOut` source of VClamp to `injectMsg`

   dest of Compartment. Connect the `VmOut` source of Compartment to

   `set_sensed` dest of VClamp. Either set `command` field to a

   fixed value, or connect an appropriate source of command potential

   (like the `outputOut` message of an appropriately configured

   PulseGen) to `set_command` dest.

    The default settings for the RC filter and PID controller should be

   

   time constant of RC filter, tau = 5 * dt

   proportional gain of PID, gain = Cm/dt where Cm is the membrane

   	capacitance of the compartment

   integration time of PID, ti = dt

   derivative time  of PID, td = 0

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process messages from the scheduler

   .. py:method:: getCommand

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getCurrent

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getSensed

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setMode

       (*destination message field*) Assigns field value.

   .. py:method:: getMode

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTi

       (*destination message field*) Assigns field value.

   .. py:method:: getTi

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTd

       (*destination message field*) Assigns field value.

   .. py:method:: getTd

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau

       (*destination message field*) Assigns field value.

   .. py:method:: getTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGain

       (*destination message field*) Assigns field value.

   .. py:method:: getGain

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: sensedIn

       (*destination message field*)  The `VmOut` message of the Compartment object should be connected
 here.

   .. py:method:: commandIn

       (*destination message field*)   The command voltage source should be connected to this.

   .. py:method:: process

       (*destination message field*) Handles 'process' call on each time step.

   .. py:method:: reinit

       (*destination message field*) Handles 'reinit' call

   .. py:attribute:: currentOut

      double (*source message field*) Sends out current output of the clamping circuit. This should be connected to the `injectMsg` field of a compartment to voltage clamp it.

   .. py:attribute:: command

      double (*value field*) Command input received by the clamp circuit.

   .. py:attribute:: current

      double (*value field*) The amount of current injected by the clamp into the membrane.

   .. py:attribute:: sensed

      double (*value field*) Membrane potential read from compartment.

   .. py:attribute:: mode

      unsigned int (*value field*) Working mode of the PID controller.
mode = 0, standard PID with proportional, integral and derivative
	all acting on the error.
mode = 1, derivative action based on command input
mode = 2, proportional action and derivative action are based on
command input.

   .. py:attribute:: ti

      double (*value field*) Integration time of the PID controller. Defaults to 1e9, i.e. integral
action is negligibly small.

   .. py:attribute:: td

      double (*value field*) Derivative time of the PID controller. This defaults to 0,
i.e. derivative action is unused.

   .. py:attribute:: tau

      double (*value field*) Time constant of the lowpass filter at input of the PID
controller. This smooths out abrupt changes in the input. Set it to 
5 * dt or more to avoid overshoots.

   .. py:attribute:: gain

      double (*value field*) Proportional gain of the PID controller.

.. py:class:: VectorTable

   .. py:method:: setXdivs

       (*destination message field*) Assigns field value.

   .. py:method:: getXdivs

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmin

       (*destination message field*) Assigns field value.

   .. py:method:: getXmin

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXmax

       (*destination message field*) Assigns field value.

   .. py:method:: getXmax

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getInvdx

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTable

       (*destination message field*) Assigns field value.

   .. py:method:: getTable

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getLookupvalue

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getLookupindex

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:attribute:: xdivs

      unsigned int (*value field*) Number of divisions.

   .. py:attribute:: xmin

      double (*value field*) Minimum value in table.

   .. py:attribute:: xmax

      double (*value field*) Maximum value in table.

   .. py:attribute:: invdx

      double (*value field*) Maximum value in table.

   .. py:attribute:: table

      vector<double> (*value field*) The lookup table.

   .. py:attribute:: lookupvalue

      double,double (*lookup field*) Lookup function that performs interpolation to return a value.

   .. py:attribute:: lookupindex

      unsigned int,double (*lookup field*) Lookup function that returns value by index.

.. py:class:: ZombieBufPool

.. py:class:: ZombieCaConc

   .. py:attribute:: proc

      void (*shared message field*) Shared message to receive Process message from scheduler

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setCa

       (*destination message field*) Assigns field value.

   .. py:method:: getCa

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCaBasal

       (*destination message field*) Assigns field value.

   .. py:method:: getCaBasal

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCa_base

       (*destination message field*) Assigns field value.

   .. py:method:: getCa_base

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setTau

       (*destination message field*) Assigns field value.

   .. py:method:: getTau

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setB

       (*destination message field*) Assigns field value.

   .. py:method:: getB

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setThick

       (*destination message field*) Assigns field value.

   .. py:method:: getThick

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setCeiling

       (*destination message field*) Assigns field value.

   .. py:method:: getCeiling

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setFloor

       (*destination message field*) Assigns field value.

   .. py:method:: getFloor

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: current

       (*destination message field*) Calcium Ion current, due to be converted to conc.

   .. py:method:: currentFraction

       (*destination message field*) Fraction of total Ion current, that is carried by Ca2+.

   .. py:method:: increase

       (*destination message field*) Any input current that increases the concentration.

   .. py:method:: decrease

       (*destination message field*) Any input current that decreases the concentration.

   .. py:method:: basal

       (*destination message field*) Synonym for assignment of basal conc.

   .. py:attribute:: concOut

      double (*source message field*) Concentration of Ca in pool

   .. py:attribute:: Ca

      double (*value field*) Calcium concentration.

   .. py:attribute:: CaBasal

      double (*value field*) Basal Calcium concentration.

   .. py:attribute:: Ca_base

      double (*value field*) Basal Calcium concentration, synonym for CaBasal

   .. py:attribute:: tau

      double (*value field*) Settling time for Ca concentration

   .. py:attribute:: B

      double (*value field*) Volume scaling factor

   .. py:attribute:: thick

      double (*value field*) Thickness of Ca shell.

   .. py:attribute:: ceiling

      double (*value field*) Ceiling value for Ca concentration. If Ca > ceiling, Ca = ceiling. If ceiling <= 0.0, there is no upper limit on Ca concentration value.

   .. py:attribute:: floor

      double (*value field*) Floor value for Ca concentration. If Ca < floor, Ca = floor

.. py:class:: ZombieCompartment

   Compartment object, for branching neuron models.

.. py:class:: ZombieEnz

.. py:class:: ZombieFuncPool

   .. py:method:: input

       (*destination message field*) Handles input to control value of n_

.. py:class:: ZombieHHChannel

   .. py:attribute:: proc

      void (*shared message field*) This is a shared message to receive Process message from thescheduler. The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which holds lots of information about current time, thread, dt andso on.
 The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo.

   .. py:method:: process

       (*destination message field*) Handles process call

   .. py:method:: reinit

       (*destination message field*) Handles reinit call

   .. py:method:: setGbar

       (*destination message field*) Assigns field value.

   .. py:method:: getGbar

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setEk

       (*destination message field*) Assigns field value.

   .. py:method:: getEk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setGk

       (*destination message field*) Assigns field value.

   .. py:method:: getGk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: getIk

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setXpower

       (*destination message field*) Assigns field value.

   .. py:method:: getXpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setYpower

       (*destination message field*) Assigns field value.

   .. py:method:: getYpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZpower

       (*destination message field*) Assigns field value.

   .. py:method:: getZpower

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setInstant

       (*destination message field*) Assigns field value.

   .. py:method:: getInstant

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setX

       (*destination message field*) Assigns field value.

   .. py:method:: getX

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setY

       (*destination message field*) Assigns field value.

   .. py:method:: getY

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setZ

       (*destination message field*) Assigns field value.

   .. py:method:: getZ

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: setUseConcentration

       (*destination message field*) Assigns field value.

   .. py:method:: getUseConcentration

       (*destination message field*) Requests field value. The requesting Element must provide a handler for the returned value.

   .. py:method:: concen

       (*destination message field*) Incoming message from Concen object to specific conc to usein the Z gate calculations

   .. py:method:: createGate

       (*destination message field*) Function to create specified gate.Argument: Gate type [X Y Z]

   .. py:method:: setNumGateX

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateX

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateY

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateY

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:method:: setNumGateZ

       (*destination message field*) Assigns number of field entries in field array.

   .. py:method:: getNumGateZ

       (*destination message field*) Requests number of field entries in field array.The requesting Element must provide a handler for the returned value.

   .. py:attribute:: Gbar

      double (*value field*) Maximal channel conductance

   .. py:attribute:: Ek

      double (*value field*) Reversal potential of channel

   .. py:attribute:: Gk

      double (*value field*) Channel conductance variable

   .. py:attribute:: Ik

      double (*value field*) Channel current variable

   .. py:attribute:: Xpower

      double (*value field*) Power for X gate

   .. py:attribute:: Ypower

      double (*value field*) Power for Y gate

   .. py:attribute:: Zpower

      double (*value field*) Power for Z gate

   .. py:attribute:: instant

      int (*value field*) Bitmapped flag: bit 0 = Xgate, bit 1 = Ygate, bit 2 = ZgateWhen true, specifies that the lookup table value should beused directly as the state of the channel, rather than usedas a rate term for numerical integration for the state

   .. py:attribute:: X

      double (*value field*) State variable for X gate

   .. py:attribute:: Y

      double (*value field*) State variable for Y gate

   .. py:attribute:: Z

      double (*value field*) State variable for Y gate

   .. py:attribute:: useConcentration

      int (*value field*) Flag: when true, use concentration message rather than Vm tocontrol Z gate

.. py:class:: ZombieMMenz

.. py:class:: ZombiePool

.. py:class:: ZombieReac

.. py:class:: testSched

   .. py:method:: process

       (*destination message field*) handles process call

