REPO_PATH := $(shell git rev-parse --show-toplevel)

PYTHON := $(REPO_PATH)/.venv/bin/python3
CXX := c++

MODULE_NAME := msep
SRC := msep.cpp

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

EIGEN_INCLUDE := $(REPO_PATH)/.eigen

CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC
CXXFLAGS += -I$(EIGEN_INCLUDE)

UNAME_S := $(shell uname -s)
LDFLAGS :=

ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -undefined dynamic_lookup
endif

TARGET := $(MODULE_NAME)$(EXT_SUFFIX)

all:
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)