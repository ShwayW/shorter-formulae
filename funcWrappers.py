# The funcWrappers module contains wrapper functions for the C++ functions
# Other modules may import this module to call C++ function easily

from ctypes import byref, c_double, c_int64, c_char_p, CDLL, POINTER
import os
import numpy as np

# to compile all available C++ functions for the project:
# g++ -std=c++20 -O2 -fPIC -shared -o extGetHeadAndInputs.so cpp/extGetHeadAndInputs.cpp && g++ -std=c++20 -O2 -fPIC -shared -o extGetSubfmlByInd.so cpp/extGetSubfmlByInd.cpp && g++ -std=c++20 -O2 -fPIC -shared -o extEvalPN.so cpp/extEvalPN.cpp

def wrapExtEvalPN(funcStr, inputs):
    # handle the edge case
    if (funcStr == ''): return 0, 1 # 1 stack left indicating error

    # formula validity check
    assert "(" not in funcStr and ")" not in funcStr, "funcStr must be in polish notation"

    # convert a number to a list if for rebustness of code
    if (not isinstance(inputs, np.ndarray)):
        if (not isinstance(inputs, list)):
            inputs = [[inputs]]

        # [1, 2, 3,...] ==> [[1], [2], [3],...]
        for i in range(len(inputs)):
            if (not isinstance(inputs[i], list)):
                inputs[i] = [inputs[i]]

    # convert all sublists to the numpy type
    inputs = np.ascontiguousarray(inputs, dtype=np.float64)

    # create a ctypes array of pointers
    cinputs = (len(inputs) * POINTER(c_double))()
    for i in range(len(inputs)):
        cinputs[i] = inputs[i].ctypes.data_as(POINTER(c_double))

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    csize = c_int64(len(inputs))
    outputs = np.zeros((len(inputs),)).astype(np.float64)
    coutputs = outputs.ctypes.data_as(POINTER(c_double))
    stacklefts = np.zeros((len(inputs),)).astype(np.float64)
    cstacklefts = stacklefts.ctypes.data_as(POINTER(c_double))

    # build the callable from C++
    [extEvalPN, free_mem] = _build_extEvalPN()

    # call the extParseEval function
    extEvalPN(byref(cfuncStr), cinputs, byref(csize), byref(coutputs), byref(cstacklefts))

    # convert results to python types
    outputs = np.ctypeslib.as_array(coutputs, shape = (len(inputs),))
    #outputs = np.nan_to_num(outputs, nan = 0.0) # nans are converted to 0.0
    stacklefts = np.ctypeslib.as_array(cstacklefts, shape = (len(inputs),))
    outputs_copy = outputs.copy()
    stacklefts_copy = stacklefts.copy()

    # free memory
    free_mem(cfuncStr, cinputs, csize, coutputs, cstacklefts)

    # return the value
    return outputs_copy, stacklefts_copy


def wrapExtGetHeadAndInputs(funcStr):
    # build the callable from C++
    [extGetHeadAndInputs, free_mem] = _build_extGetHeadAndInputs()

    # convert the inputs into correcponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    chead = c_char_p()
    cinputs = POINTER(c_char_p)()
    csize = c_int64()

    # call the extGetSubfmlByInd function
    extGetHeadAndInputs(byref(cfuncStr), byref(chead), byref(cinputs), byref(csize))
    head = chead.value.decode("utf-8")
    inputs = [cinputs[i].decode("utf-8") for i in range(csize.value)]

    # free memory
    free_mem(chead, cinputs, csize)

    # return the value
    return [head, inputs]


#################### Private Functions ###################
# The builder functions for building compiled C++ shared libraries

# Resolve .so paths relative to this file so callers work from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Module-level caches so each worker process (or the main process) pays the
# CDLL construction cost only once instead of on every wrapExt* call.
_LIB_EVAL_CACHED = None
_LIB_HEAD_AND_INPUTS_CACHED = None

def _build_extEvalPN():
    global _LIB_EVAL_CACHED
    if _LIB_EVAL_CACHED is None:
        lib = CDLL(os.path.join(_HERE, "extEvalPN.so"))
        lib.extEvalPN.argtypes = (POINTER(c_char_p), POINTER(POINTER(c_double)), POINTER(c_int64), POINTER(POINTER(c_double)), POINTER(POINTER(c_double)))
        lib.extEvalPN.restype = None
        lib.free_mem.argtypes = (c_char_p, POINTER(POINTER(c_double)), c_int64, POINTER(c_double), POINTER(c_double))
        lib.free_mem.restype = None
        _LIB_EVAL_CACHED = [lib.extEvalPN, lib.free_mem]
    return _LIB_EVAL_CACHED


def _build_extGetHeadAndInputs():
    global _LIB_HEAD_AND_INPUTS_CACHED
    if _LIB_HEAD_AND_INPUTS_CACHED is None:
        lib = CDLL(os.path.join(_HERE, "extGetHeadAndInputs.so"))
        lib.extGetHeadAndInputs.argtypes = (POINTER(c_char_p), POINTER(c_char_p), POINTER(POINTER(c_char_p)), POINTER(c_int64))
        lib.extGetHeadAndInputs.restype = None
        lib.free_mem.argtypes = (c_char_p, POINTER(c_char_p), c_int64)
        lib.free_mem.restype = None
        _LIB_HEAD_AND_INPUTS_CACHED = [lib.extGetHeadAndInputs, lib.free_mem]
    return _LIB_HEAD_AND_INPUTS_CACHED


