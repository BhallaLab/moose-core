% MOOSEGUI: Graphical interface for MOOSE
% H. Chaitanya; Harsha Rani; Subhasis Ray; Upi Bhalla
% May 11, 2012

# Introduction

The Moose GUI lets you work on [chemical](Kkit12Documentation.html) and 
[compartmental neuronal](Nkit2Documentation.html) models using a common 
interface framework. This document describes this common framework. In MOOSE 
2.0.0, the interface lets you read, run, edit, and write chemical kinetic 
models, and to read, edit and run neuronal models.

# Layout of interface

The **MooseGui** interface consists of a model view window to the left, 
occupying most of the screen. To the right there is a panel with controls 
for viewing and editing model parameters, for configuring plots, and for 
running the model.

![](../../images/MooseGuiImage.png)

# The menu bar

In Ubuntu 12.04, the menu bar appears only when the mouse is in the top menu 
strip of the screen. In other distributions it should appear over the top of 
the interface.

![](../../images/MooseGuiMenuImage.png)

The menu bar contains the following entries: **`File`**, **`Edit`**, **`View`**, 
**`Solver`**, **`Help`**.

##File

-   **Load Model**: This leads to a file finder dialog. The default location 
    should bring you to the `Demos` directory of MOOSE. Here you can go into 
    the `Genesis-files` subdirectory, which contains legacy Kinetikit (`*.g`) 
    format models. The `neuroml` subdirectory leads to a number of example 
    neuronal and network models, each in their own subdirectory. The other 
    demos are mostly standalone demos.
-   **Save Model**: This currently works only for chemical kinetic models, and 
    only works for the Kinetikit format.
-   **Merge Models**: This allows one to load in a new chemical kinetic model 
    on top of an already loaded model.
-   **Save plots**: Dumps plot contents to disk in xplot (ascii) format.
-   **Shell model**: Not yet operational.
-   **Reset settings**: Restores interface settings to default.
-   **Quit**: Quits the interface.

##Edit

-   **Settings**: Provides for changing interface settings such as file 
locations.

##View

This allows one to control display of various parts of the
interface.

-   **Moose shell**: Currently inactive.
-   **Property editor**: Toggles visibility of the panels for viewing and 
    editing object parameters. This is to the upper right of the interface in 
    the interface layout example figure above.
-   **Simulation control**: Toggles visibility of the panels for controlling 
    running of the simulation, at the bottom right of the interface layout.
-   **Plot config**: Toggles visibility of panels for making plots. This is in 
    the middle right of the interface layout.
-   **Sub Windows**: This sets up the model view and the plot view panels into 
    separate sub-windows of the screen. Useful when you want to watch plots at 
    the same time as the cell display, or to select and edit different parts of 
    the model while watching the simulation progress.
-    **Tabs**: This sets up the model view and plot view panels into tabs on 
    the screen, as in the example interface figure above. This is useful to 
    dedicate a larger portion of screen area to each display.

##Solver

The Solver options currently only select between methods for kinetic models. 
Details are in the [chemical kinetics 
documentation](Kkit12Documentation.html).

Options are:

-   **Runge Kutta**: This is the default method for integration of ODE systems. It 
    uses the Gnu Scientific Library 5-th order explicit variable-timestep 
    Runge-Kutta-Fehlberg method.
-   **Gillespie**: This is an implementation of Gillespies Stochastic Systems 
    Algorithm, which computes reaction progress using a stochastic method.

##Help
-   **About**: Version and general information about MOOSE.
-   **General documentation**: This file. How to use the common interface 
    framework.
-   **Kkit**: Documentation on the use of Kinetikit version 12.
-   **Nkit**: Documentation on the use of Neurokit version 2.
-   **Report a bug**: Takes you to the SourceForge bug tracker for Moose.

# The Model window

The Model window displays a view of the model structure. These views vary 
depending on the model type.

## 3-D display of neuronal models in Neurokit:

![](../../images/NkitModelWindow.png)

The individual compartments of the neuron model can be clicked to select, 
and when selected, the compartment parameters and variables are displayed in 
the **`Property editor`** described below. For a neuronal or neuronal network 
model, the window displays a 3-D view of the cell(s) in the model. It does 
so using **OpenGL**, which is a standard for displaying 3-D views. In 
addition, the display sets the color of each compartment based on some 
variable value, typically `Vm`, the membrane potential of the compartment. 
Note the 3-D axis indicators in the bottom left.

The controls for moving the display are as follows:

-   **Zoom**: One can zoom in and out of the field of view in the OpenGL 
    window using the comma and period keys (think in terms of the angle bracket 
    symbols on the same keys). In addition, the scroll wheel or the vertical 
    scroll line on the track pad will also cause the display to zoom in and out.
-   **Pan**: The arrow keys will move the display left, right, up and down. 
    Pan can also be done using the mouse if you hold down the **`shift`** key and 
    the **`left mouse button`** at the same time.
-   **Pitch**: For reasons entirely unknown to me, the keys **`y`** and **`u`** 
    are assigned to rotate the display about the vertical axis, otherwise known 
    as *pitch*.
-   **Yaw**: The keys **`q`** and **`a`** control rotation around the horizontal 
    axis, otherwise known as *yaw*.
-   **Roll**: To complete the key binding hall of fame, the keys **`z`** and 
    **`x`** rotate the display around the axis going into the display from your 
    eyes.
-   **Mouse controls**: One can also acheive a combination of *pitch*, *roll*, 
    and *yaw* by holding the **`left mouse button`** down and moving the mouse.

## 2-D display of chemical kinetics models in Kinetikit:

For a chemical kinetics network, the window displays a schematic of
the chemical reaction system. This is in the tab labeled
**`Kkit Layout`**. There are distinct icons for molecules, reactions
and enzymes, and these are connected by arrows to set up the
reaction scheme. Again, any icon can be clicked to select and its
parameters and variables come up in the **`Property editor`**.

![](../../images/KkitModelWindow.png)

The chemical network is displayed only in 2 dimensions. The
controls are correspondingly simpler:

-   **Zoom**: Comma and period keys. Alternatively, the mouse scroll wheel or 
    vertical scroll line on the track pad will cause the display to zoom in and 
    out.
-   **Pan**: The arrow keys move the display left, right, up, and down.
-   **Entire Model View**: Pressing the **`a`** key will fit the entire model 
    into the entire field of view.
-   **Resize Icons**: Angle bracket keys, that is, **`<`** and **`>`**. This 
    resizes the icons while leaving their positions on the screen layout more or 
    less the same.
-   **Original Model View**: Presing the **`A`** key (capital `A`) will revert to 
    the original model view including the original icon scaling.

# The plot window

The plot window displays time-series plots of the simulation. Plots are 
color-coded to distinguish them. In the case of the **kkit** interface the 
plots take the same color as the molecule pool that they represent.

![](../../images/KkitPlotWindow.png)

The plots are done using **MatPlotLib**, so the usual controls apply. Beneath 
the plot window there is a little row of icons:

![](../../images/PlotWindowIcons.png)

These are the plot controls. If you hover the mouse over them for a few 
seconds, a function reminder box pops up. The functions as follows:

![](../../images/MatPlotLibHomeIcon.png)

-   **Home**: Returns the plot display to its default position.

![](../../images/MatPlotLibDoUndo.png)

-   **Undo/Redo**: Undoes or re-does manipulations you have done to
the display.

![](../../images/MatPlotLibPan.png)

-   **Pan**: The plots will pan around with the mouse when you hold the left 
button down. The plots will zoom with the mouse when you hold the right 
button down.

![](../../images/MatPlotLibZoom.png)

-   **Zoom to rectangle**: With the **`left mouse button`**, this
    will zoom in to the specified rectangle so that the plots become
    bigger. With the **`right mouse button`**, the entire plot display
    will be shrunk to fit into the specified rectangle.

![](../../images/MatPlotLibConfigureSubplots.png)

-   **Configure subplots**: You don't want to mess with these.

![](../../images/MatPlotLibSave.png)

-   **Save**: Pops up a dialog box to save the plot. At this point
    it only saves into a `.png` file.

# The side panel

The **`side panel`** is located on the right of the screen. It displays
three controls: the **`Property editor`**, the **`Plot configuration`**
and the **`Run control`**.

## Property editor

The **`Property editor`** displays parameters and variables of the
selected model component (object).

![](../../images/PropertyEditor.png)

The object many be a compartments of a neuronal model, or pools,
reactions, or enzymes in a signaling model.

-   **The top of the property editor**: displays the class and path
    of the selected object.
-   **See children**: opens a subsidiary table to navigate to
    child objects in the filesystem-like object tree.
-   **Select Parent**: Navigates back up to the parent object in
    the element tree.
-   **Properties**: This table displays field names in the first column,
    followed by field values in the second. If the field is editable one
    can click on the value in the second column and change it.

## Plot configuration

The **`Plot configuration`** panel lets one set up new plots based on
selected objects and their fields.

![](../../images/PlotConfig.png)

-   **The top line**: has the name of the object whose field is to be plotted.
-   **Plot Field**: Specifies field to be plotted.
-   **Plot Window**: Specifies which of the existing plot windows to use for
    the new plot.
-   **New Plot Tab**: This button creates a new plot window as a tab.
-   **Add Field**: Creates the plot as specified by the other options.
-   **Overlay Plots**: When not checked, plots are cleared every time the
    `Reset` button is hit. When checked, this retains the plots from the
    previous run.

## Simulation control

The **`Simulation control`** panel controls how the model is run.

![](../../images/SimulationControl.png)

-   **Run Time**: Determines duration for which simulation is to
    run. If simulation has already run, this runs for the specified
    additional period.
-   **Reset**: Restores simulation to its initial state;
    reinitializes all variables to t = 0.
-   **Stop**: This button halts an ongoing simulation.
-   **Current time**: This reports the current simulation time.
-   **Advanced options**: This is available only after `Reset`.
    This sets:
    -   **Plotdt**: Timestep to use for updating plots.
    -   **Simdt**: Timestep to use for internal simulation clocks. Edit
        only if you know what you are doing. For kinetic models, most of
        the numerical methods use variable timestep calculations, so this
        should be set to the same value as the `Plotdt` in most cases.
    -   **Update Plotdt**: How frequently should the screen refresh.
