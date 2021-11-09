#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
//#include <stdio.h>



__device__ void init_invert_page_table(VirtualMemory *vm) {
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        INIT(vm,i);
        SET_INVALID(vm,i);
        SET_PREV_NULL(vm,i);
        SET_NEXT_NULL(vm,i);
    }

    for (int i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE);i++)
        INIT_DISK_MAP(vm,i);
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
    for(u32 i=0;i<vm->PAGESIZE;i++)
       vm->buffer[(mem_entry<<5|(i&0x1F))] = vm->storage[(disk_entry<<5|(i&0x1F))];
    UNSET_DIRTY(vm,mem_entry); // fresh entry certainly not dirty
}

__device__ void swap_out(VirtualMemory *vm,u32 mem_entry, u32 disk_entry, u32 vm_addr, u32 pid) {
    UNSET_DISK_INVALID(vm,disk_entry);
    SET_DISK_TO_VM(vm,disk_entry,pid,vm_addr);
    for(u32 i=0;i<vm->PAGESIZE;i++)
      vm->storage[(disk_entry<<5|(i&0x1F))] = vm->buffer[(mem_entry<<5|(i&0x1F))];
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
    if(IS_DIRTY(vm,target)){
      for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break;
    }
    //printf("SWAPPED OUT MEM #%d to DISK #%d",target,disk_entry);
    swap_out(vm,target,disk_entry,vm_page_addr);
    }
    
    return target;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr, Lock* lock_ptr, u32 pid) {
  /* Complate vm_read function to read single element from data buffer */
  for(int i=3; i>pid; i--) lock_ptr->try_lock(i);
  for(int i = pid;i>=0;i--) lock_ptr->lock(i);
  lock_ptr->lock(DATA_LOCK);
  for(int i = 0; i<DATA_LOCK; i++) lock_ptr->unlock(i);
  u32 vm_page_addr = (addr >> 5);
  u32 vm_page_offset = addr & 0x1F;
  if(GET_COUNT(vm) < vm->PAGE_ENTRIES) {
    /** non-full case, use hashing **/
    /** no need to consider replacement **/
    u32 mem_entry; // this can also mark the first empty slot appearing
    for(u32 i=0; i < vm->PAGE_ENTRIES; i++) {
      mem_entry = HASH(vm_page_addr,i);
      if(IS_INVALID(vm,mem_entry)) break;
      if(GET_ADDR(vm,mem_entry) == vm_page_addr && GET_PID(vm,mem_entry) == pid) {
        mark_use(vm,mem_entry);
        lock_ptr->unlock(DATA_LOCK);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }

    }
    /** unable to find in mem, page fault check disk **/
    (*(vm->pagefault_num_ptr))++;
    for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      u32 disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break; //cannot found in disk either, error
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr && GET_PID_FROM_DISK(vm,disk_entry)==pid) {
        /* page allocation routine */
        UNSET_INVALID(vm,mem_entry);
        SET_ADDR(vm,mem_entry,vm_page_addr,pid);
        SET_COUNT(vm,GET_COUNT(vm)+1);
        /* swap_in and mark use */
        swap_in(vm,mem_entry,disk_entry);
        mark_use(vm,mem_entry);
        //return;
        lock_ptr->unlock(DATA_LOCK);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }

  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(IS_INVALID(vm,mem_entry)==0 && GET_ADDR(vm,mem_entry) == vm_page_addr && GET_PID(vm,mem_entry)==pid){
        mark_use(vm,mem_entry);
        //printf("Page found: #%d\n",mem_entry);
        lock_ptr->unlock(DATA_LOCK);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }
    
    /** unable to find in mem, page fault and check disk **/
    (*(vm->pagefault_num_ptr))++;
    for(u32 i=0; i < (vm->STORAGE_SIZE/vm->PAGESIZE); i++) {
      u32 disk_entry = HASH_DISK(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break;
      //printf("DISK_VM: #%d MY_VM: #%d\n",GET_VM_FROM_DISK(vm,disk_entry),vm_page_addr);
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr && GET_PID_FROM_DISK(vm,disk_entry)==pid) {
        //printf("Found on disk page: #%d\n",disk_entry);
        u32 destination = evict_lru(vm);
        /* page allocation routine */
        UNSET_INVALID(vm,destination);
        SET_ADDR(vm,destination,vm_page_addr,pid);
        // SET_COUNT(vm,GET_COUNT(vm)+1);
        // COUNT ALREADY MAX

        /* swap in and mark use */
        swap_in(vm,destination,disk_entry);
        mark_use(vm,destination);
        //return;
        lock_ptr->unlock(DATA_LOCK);
        return vm->buffer[((destination<<5) | vm_page_offset)];
      }
    }
  
  

  }
  lock_ptr->unlock(DATA_LOCK);
 return 1;
}


__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value, Lock* lock_ptr, u32 pid) {

  /* complete to aquire the data lock */
  for(int i=3; i>pid; i--) lock_ptr->try_lock(i);
  for(int i = pid;i>=0;i--) lock_ptr->lock(i);
  lock_ptr->lock(DATA_LOCK);
  for(int i = 0; i<DATA_LOCK; i++) lock_ptr->unlock(i);
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
      if(GET_ADDR(vm,mem_entry) == vm_page_addr && GET_PID(vm,mem_entry)==pid) {
        mark_use(vm,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        SET_DIRTY(vm,mem_entry);
        lock_ptr->unlock(DATA_LOCK);
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
    SET_ADDR(vm,mem_entry,vm_page_addr,pid);
    vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
    mark_use(vm,mem_entry);
  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(GET_ADDR(vm,mem_entry) == vm_page_addr && GET_PID(vm,mem_entry)==pid){
        SET_DIRTY(vm,mem_entry);
        mark_use(vm,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        lock_ptr->unlock(DATA_LOCK);
        return;
      }
    }
    
    /** unable to find in mem, page fault, evict a victim and place it there **/
    (*(vm->pagefault_num_ptr))++;
    u32 destination = evict_lru(vm);
    //printf("Evict vitcim: #%d\n",destination);
    UNSET_INVALID(vm,destination);
    /** record new vm mapping **/
    SET_ADDR(vm,destination,vm_page_addr,pid);
    SET_DIRTY(vm,destination);
    mark_use(vm,destination);
    vm->buffer[((destination<<5) | vm_page_offset)] = value;
    lock_ptr->unlock(DATA_LOCK);
    return;

 }
 lock_ptr->unlock(DATA_LOCK);
}


__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
   for(u32 i=0;i<input_size;i++)
      results[i] = vm_read(vm,offset+i);
}