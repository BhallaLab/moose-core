# Author: Subhasis Ray
# Date: Sun Jul  7

builtins_src = ['Arith.cpp',
                'Group.cpp',
                'Mstring.cpp',
                'Function.cpp',
                'Variable.cpp',
                'InputVariable.cpp',
                'TableBase.cpp',
                'Table.cpp',
                'Interpol.cpp',
                'StimulusTable.cpp',
                'TimeTable.cpp',
                'StreamerBase.cpp',
                'Streamer.cpp',
                'Stats.cpp',
                'Interpol2D.cpp',
                'SpikeStats.cpp',
                'MooseParser.cpp',
                '../utility/cnpy.cpp',
                '../external/fmt/src/format.cc',
                '../external/fmt/src/os.cc',
                'testBuiltins.cpp']

if host_machine.system() != 'windows'
  builtins_src += files(['SocketStreamer.cpp'])
endif

builtins_lib = static_library('builtins', builtins_src, include_directories: ['../external/fmt/include'])
