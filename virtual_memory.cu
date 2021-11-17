#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ u32 hash(u32 x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}
__device__ void init_invert_page_table(VirtualMemory *vm) {
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        INIT(vm,i);
        SET_INVALID(vm,i);
        SET_PREV_NULL(vm,i);
        SET_NEXT_NULL(vm,i);
    }

    for (int i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE);i++) {
      INIT_DISK_MAP(vm,i);
      SET_DISK_INVALID(vm,i);
    }
    SET_COUNT(vm,0);
    SET_EMPTY(vm);

}
/** Usage:
  vm_init(&vm, data, storage,
          pt, &pagefault_num,
          PAGE_SIZE, INVERT_PAGE_TABLE_SIZE,
          PHYSICAL_MEM_SIZE, STORAGE_SIZE,
          PHYSICAL_MEM_SIZE / PAGE_SIZE);
**/
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ void mark_use(VirtualMemory *vm, u32 mem_entry) {
    /** MOVE MEM_ENTRY TO HEAD IN LINKED LIST **/
    if(IS_EMPTY(vm)) {
        SET_HEAD_NODE(vm,mem_entry);
        SET_TAIL_NODE(vm,mem_entry);
        UNSET_EMPTY(vm);
    }
    /** DELETE NODE FROM LIST **/
    if(GET_HEAD_NODE(vm)!=mem_entry){
        
        if(NEXT_IS_NULL(vm,mem_entry)==0){
        if(PREV_IS_NULL(vm,mem_entry)==0){
            SET_PREV(vm,GET_NEXT(vm,mem_entry),GET_PREV(vm,mem_entry));
        }
        else{
            SET_PREV_NULL(vm,GET_NEXT(vm,mem_entry));
        }
    }
    
    if(PREV_IS_NULL(vm,mem_entry)==0){
        if(NEXT_IS_NULL(vm,mem_entry)==0) {
            
            SET_NEXT(vm,GET_PREV(vm,mem_entry),GET_NEXT(vm,mem_entry));
        }
        else {
            SET_NEXT_NULL(vm,GET_PREV(vm,mem_entry));
        }
            
    }
    
    u32 old_head = GET_HEAD_NODE(vm);
    SET_NEXT(vm,mem_entry,old_head);
    SET_PREV(vm,old_head,mem_entry);
    SET_HEAD_NODE(vm,mem_entry);
        
        
    }
    
    
}

__device__ void swap_in(VirtualMemory *vm,u32 mem_entry, u32 disk_entry) {
    //printf("[SWAP IN] FROM DISK PAGE#%d to MEM PAGE#%d VM=%d\n",disk_entry,mem_entry,GET_VM_FROM_DISK(vm,disk_entry));
    for(u32 i=0;i<vm->PAGESIZE;i++)
       vm->buffer[(mem_entry<<5)+i] = vm->storage[(disk_entry<<5)+i];
    UNSET_DIRTY(vm,mem_entry); // fresh entry certainly not dirty
}

__device__ void swap_out(VirtualMemory *vm,u32 mem_entry, u32 vm_addr) {
   u32 disk_entry;
   u32 i;
    for(i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      disk_entry = HASH_DISK(vm_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)!=0) break;
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_addr) break;
    }
    UNSET_DISK_INVALID(vm,disk_entry);
    SET_DISK_TO_VM(vm,disk_entry,0,vm_addr);
    // if(i==(vm->STORAGE_SIZE/vm->PAGESIZE)) printf("[ERROR] VM=%d",vm_addr);
    // printf("[SWAP OUT] FROM MEM PAGE#%d to DISK PAGE#%d \n VM=%d",mem_entry,disk_entry,vm_addr);
    for(u32 i=0;i<vm->PAGESIZE;i++)
      vm->storage[(disk_entry<<5)+i] = vm->buffer[(mem_entry<<5)+i];
}

__device__ u32 extract_lru(VirtualMemory *vm) {
    u32 target = GET_TAIL_NODE(vm);
    if(PREV_IS_NULL(vm,target)==0) {
      /** target not head, therefore has prev **/
      u32 new_tail = GET_PREV(vm,target);
      SET_TAIL_NODE(vm,new_tail);
    }
    else SET_EMPTY(vm);
    return target;
}


__device__ u32 evict_lru(VirtualMemory *vm) {
    u32 target = extract_lru(vm);
    u32 disk_entry;
    u32 vm_page_addr = GET_ADDR(vm,target);
    u32 i;
    if(IS_DIRTY(vm,target)){
    swap_out(vm,target,vm_page_addr);
    //if(DISK_IS_INVALID(vm,disk_entry)==0) printf("ERROR ON DISK ENTRY %d; i=%d\n",disk_entry,i);
    //printf("SWAPPED OUT MEM #%d to DISK #%d",target,disk_entry);
    //swap_out(vm,target,disk_entry,vm_page_addr);
    }
    
    return target;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */

  u32 vm_page_addr = (addr >> 5);
  u32 vm_page_offset = addr & 0x1F;
  if(GET_COUNT(vm) < vm->PAGE_ENTRIES) {
    /** non-full case, use hashing **/
    /** no need to consider replacement **/
    u32 mem_entry; // this can also mark the first empty slot appearing
    for(u32 i=0; i < vm->PAGE_ENTRIES; i++) {
      mem_entry = HASH(vm_page_addr,i);
      if(IS_INVALID(vm,mem_entry)) break;
      if(GET_ADDR(vm,mem_entry) == vm_page_addr) {
        mark_use(vm,mem_entry);
        //printf("VM[%d]: Found and read mem page: #%d\n",addr,mem_entry);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }

    }
    /** unable to find in mem, page fault check disk **/
    (*(vm->pagefault_num_ptr))++;
    for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      u32 disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break; //cannot found in disk either, error
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr) {
        /* page allocation routine */
        UNSET_INVALID(vm,mem_entry);
        SET_ADDR(vm,mem_entry,vm_page_addr);
        SET_COUNT(vm,GET_COUNT(vm)+1);
        /* swap_in and mark use */
        //printf("VM[%d]: Unfound in mem and swap from disk %d to mem %d\n",addr,disk_entry,mem_entry);
        swap_in(vm,mem_entry,disk_entry);
        mark_use(vm,mem_entry);
        //return;
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }

  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(IS_INVALID(vm,mem_entry)==0 && GET_ADDR(vm,mem_entry) == vm_page_addr){
        mark_use(vm,mem_entry);
        //printf("Page found: #%d\n",mem_entry);
        //printf("VM[%d]: Found and read mem page: #%d\n",addr,mem_entry);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }
    
    /** unable to find in mem, page fault and check disk **/
    (*(vm->pagefault_num_ptr))++;
    for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      u32 disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break;
      //printf("DISK_VM: #%d MY_VM: #%d\n",GET_VM_FROM_DISK(vm,disk_entry),vm_page_addr);
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr) {
        u32 destination = evict_lru(vm);
        //printf("VM[%d]: evict mem page %d for it\n",vm_page_addr,destination);
        /* page allocation routine */
        UNSET_INVALID(vm,destination);
        SET_ADDR(vm,destination,vm_page_addr);
        // SET_COUNT(vm,GET_COUNT(vm)+1);
        // COUNT ALREADY MAX

        /* swap in and mark use */
        swap_in(vm,destination,disk_entry);
        mark_use(vm,destination);
        //return;
        //printf("VM[%d]: Unfound in mem and swap from disk %d to mem %d\n",addr,disk_entry,destination);
        return vm->buffer[((destination<<5) | vm_page_offset)];
      }
    }
  
  

  }
 return 1;
}


__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 vm_page_addr = (addr >> 5);
  u32 vm_page_offset = addr & 0x1F;
  if(GET_COUNT(vm) < vm->PAGE_ENTRIES) {
    /** non-full case, use hashing **/
    /** no need to consider replacement **/
    u32 mem_entry; // this can also mark the first empty slot appearing
    for(u32 i=0; i < vm->PAGE_ENTRIES; i++) {
      mem_entry = HASH(vm_page_addr,i);
      if(IS_INVALID(vm,mem_entry)) break;
      if(GET_ADDR(vm,mem_entry) == vm_page_addr) {
        mark_use(vm,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        //printf("VM[%d]: Already allocted and write to mem page: #%d\n",addr,mem_entry);
        SET_DIRTY(vm,mem_entry);
        return;
      }
    }
    //printf("Direct write to page#%d\n",mem_entry);
    /** no record in mem, write directly to mem_entry **/
    /** page allocation routine, page fault as well **/
    (*(vm->pagefault_num_ptr))++;
    UNSET_INVALID(vm,mem_entry);
    SET_COUNT(vm,GET_COUNT(vm)+1);
    SET_DIRTY(vm,mem_entry); // dirty at born
    /** record new vm mapping **/
    SET_ADDR(vm,mem_entry,vm_page_addr);
    //printf("VM[%d]: Allocate and write to mem page: #%d\n",addr,mem_entry);
    vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
    mark_use(vm,mem_entry);
  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(GET_ADDR(vm,mem_entry) == vm_page_addr){
        SET_DIRTY(vm,mem_entry);
        mark_use(vm,mem_entry);
        //printf("VM[%d]: Already allocted and write to mem page: #%d\n",addr,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        return;
      }
    }

    /** unable to find in mem, page fault, evict a victim and place it there **/
    (*(vm->pagefault_num_ptr))++;
    u32 destination = evict_lru(vm);
    // printf("VM[%d]: Evict vitcim: #%d\n",addr,destination);
    // printf("VM[%d]: Allocate and write to mem page: #%d\n",addr, destination);
    for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      u32 disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break; //cannot found in disk
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr) {
        /* page allocation routine */
        UNSET_INVALID(vm,destination);
        SET_ADDR(vm,destination,vm_page_addr);
        /* swap_in and mark use */
        // printf("VM[%d]: Unfound in mem and swap from disk %d to mem %d\n",addr,disk_entry,destination);
        swap_in(vm,destination,disk_entry);
        mark_use(vm,destination);
        //return;
        vm->buffer[((destination<<5) | vm_page_offset)] = value;
        SET_DIRTY(vm,destination);
        return;
      }
    }
    UNSET_INVALID(vm,destination);
    /** record new vm mapping **/
    SET_ADDR(vm,destination,vm_page_addr);
    SET_DIRTY(vm,destination);
    mark_use(vm,destination);
    vm->buffer[((destination<<5) | vm_page_offset)] = value;
    return;
 }
}


__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
   for(u32 i=0;i<input_size;i++)
      results[i] = vm_read(vm,offset+i);
}