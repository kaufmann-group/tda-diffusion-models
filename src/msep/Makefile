PYTHON := ../../.venv/bin/python3
CXX := c++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC

MODULE_NAME := msep_cpp
SRC := msep.cpp

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC
LDFLAGS := -undefined dynamic_lookup

TARGET := $(MODULE_NAME)$(EXT_SUFFIX)

all:
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)