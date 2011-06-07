/**
\mainpage MOOSE source code documentation

\section intro_sec Introduction

MOOSE is the base and numerical core for large, detailed simulations 
including Computational Neuroscience and Systems Biology. MOOSE spans the 
range from single molecules to subcellular networks, from single cells to 
neuronal networks, and to still larger systems. it is backwards-compatible 
with GENESIS, and forward compatible with Python and XML-based model 
definition standards like SBML and NeuroML. 

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

\ref DesignDocument
*/
