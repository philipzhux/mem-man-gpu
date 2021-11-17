#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  // for (int i = 0; i < input_size; i++)
  //   vm_write(vm, i, input[i]);

  // for (int i = input_size - 1; i >= input_size - 32769; i--)
  //   int value = vm_read(vm, i);

  // vm_snapshot(vm, results, 0, input_size);
  // for (int i = 0; i < input_size; i++)
  // if(i%2==0) vm_write(vm, i, input[i]);
  // for (int i = 0; i < input_size; i++)
  // if(i%2) vm_write(vm, i, input[i]);
  // for (int i = input_size - 1; i >= input_size - 32769; i--)
  // if(i%2) int value = vm_read(vm, i);

  //vm_write(vm, 0, 0x01);


  //vm_write(vm, 33, 0xFF);
  // for (int i = 0; i < input_size; i++)
  // if(i%2==0) vm_write(vm, i, input[i]);

  //vm_write(vm, 32*5, 0x1F);

  // for (int i = input_size - 32769; i >= 222 ; i--)
  // if(i%2) int value = vm_read(vm, i);
  //   for (int i = 0; i < input_size; i++)
  // if(i%2) vm_write(vm, i, input[i]);

  // vm_write(vm, 0, 0x01);

  // for (int i = input_size - 1; i >= input_size - 32769; i--)
  // if(i%2) int value = vm_read(vm, i);


  // vm_write(vm, 33, 0xFF);
  // for (int i = 0; i < input_size; i++)
  // if(i%2==0) vm_write(vm, i, input[i]);

  // vm_write(vm, 32*5, 0x1F);

  // for (int i = input_size - 32769; i >= 222 ; i--)
  // if(i%2) int value = vm_read(vm, i);

  // vm_write(vm, input_size-1, 0x17);

  // vm_snapshot(vm, results, 0, input_size);
  for (int i = 0; i < input_size; i++)
    vm_write(vm, i, input[i]);

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i);

  vm_snapshot(vm, results, 0, input_size);

}
