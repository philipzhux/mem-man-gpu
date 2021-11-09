#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size, Lock* lock_ptr, u32 pid) {
  for (int i = 0; i < input_size; i++)
    
    vm_write(vm, i, input[i],lock_ptr,pid);

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i,lock_ptr,pid);

  vm_snapshot(vm, results, 0, input_size,lock_ptr,pid);
}
