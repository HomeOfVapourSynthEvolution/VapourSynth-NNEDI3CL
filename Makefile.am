warning_flags = -Wall -Wextra -Wno-unused-parameter -Wno-ignored-attributes
common_cflags = -O3 -ffast-math -fvisibility=hidden $(warning_flags) $(MFLAGS)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS) -DNNEDI3_DATADIR='"$(pkgdatadir)"'

dist_pkgdata_DATA = NNEDI3CL/nnedi3_weights.bin

lib_LTLIBRARIES = libnnedi3cl.la

libnnedi3cl_la_SOURCES = NNEDI3CL/NNEDI3CL.cpp

libnnedi3cl_la_LDFLAGS = -no-undefined -avoid-version -lboost_filesystem -lboost_system $(PLUGINLDFLAGS) $(OPENCLLDFLAGS)
