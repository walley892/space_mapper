import pycuda.gpuarray as gpuarray
import numpy as np
_common_buffers = []
_common_buffers_inuse = []

def allocate_or_return_buffer(index, shape, dtype):
    global _common_buffers
    
    if np.prod(_common_buffers[index].shape)*_common_buffers[index].itemsize == np.prod(shape)*dtype().itemsize:
        return _common_buffers[index]
    else:
        _common_buffers[index].gpudata.free()
        _common_buffers[index] = gpuarray.zeros(np.prod(shape), dtype)
        return _common_buffers[index]

def get_buffer_for_image(shape, dtype):
    global _common_buffers
    if len(_common_buffers) == 0:
        _common_buffers.append(gpuarray.zeros(np.prod(shape), dtype))
        _common_buffers_inuse.append(True)
        return _common_buffers[0], 0
    else:
        for i, inuse in enumerate(_common_buffers_inuse):
            if not inuse:
                _common_buffers_inuse[i] = True
                return allocate_or_return_buffer(i, np.prod(shape), dtype), i
    _common_buffers.append(gpuarray.zeros(np.prod(shape), dtype))
    _common_buffers_inuse.append(True)
    return _common_buffers[-1], len(_common_buffers) - 1

def mark_buffer_free(index):
    _common_buffers_inuse[index] = False


