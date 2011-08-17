/**
\mainpage MOOSE source code documentation

\section intro_sec Introduction

MOOSE is the base and numerical core for large, detailed simulations 
including Computational Neuroscience and Systems Biology. MOOSE spans the 
range from single molecules to subcellular networks, from single cells to 
neuronal networks, and to still larger systems. it is backwards-compatible 
with GENESIS, and forward compatible with Python and XML-based model 
definition standards like SBML and NeuroML. 

MOOSE uses Python as its primary scripting language. For backward 
compatibility we have a GENESIS scripting module, but this is deprecated.
MOOSE uses Qt/OpenGL for its graphical interface. The entire GUI is
written in Python, and the MOOSE numerical code is written in C++.

MOOSE is currently in beta 1.3. One more beta, 1.4 is planned, which
will have a first-order functional GUI. Following this
we'll go into release-candidate mode with RC0.9 and so on.

\section support_sec Hardware and availability
MOOSE runs on everything from laptops to large clusters. It supports
multiple-core machines through threading, and cluster architectures using
MPI, the Message Passing Interface. MOOSE is compiled for Linux,
MacOS, and to the extent that we can get it to compile, on Windows.

MOOSE is free software.
MOOSE makes extensive use of external libraries. The main MOOSE code itself
is LGPL, meaning it is easy to reuse with attribution but will remain
free. However, the common release of MOOSE uses the GNU scientific library
(GSL) which is under the GPL. For such releases, MOOSE should be treated
as also being under the GPL.

================================================================================

Some notes about adding general documentation in Doxygen. These notes can be
removed later.

General documentation in Doxygen can be organized into "pages". A page can have
sections, subsections, and perhaps more components. Every page, section and
subsection has a label (a single-word id), a title, and body text. If the label
is unique across the entire project documentation, cross-page hyper-linked
references can be added as shown below.

Doxygen can collect all the content for a given page from multiple source files.
This is demonstrated in Docs/doxygen-programmers-guide.cpp. However, I am not
sure how will Doxygen order the sections on the page.

================================================================================

Apart from the auto-generated documentation for the source-code itself, here are
some higher-level hand-written documents:

\ref ProgrammersGuide

\ref AppProgInterface

\ref DesignDocument
*/
