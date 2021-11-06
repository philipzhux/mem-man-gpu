#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define VALID_BIT 31
#define DIRTY_BIT 30
#define GET_BIT(b,t) ((t & (0x1<<b))>>b)
#define SET_BIT(b,t) t = t | (0x1<<b)
#define UNSET_BIT(b,t) t = t & ~(0x1<<b)
#define GET_PID(vm,index) ((vm->invert_page_table[index*2]>>13) & 0x3)
#define GET_ADDR(vm,index) (vm->invert_page_table[index*2] & 0x1FFF)
#define GET_NEXT(vm,index) (vm->invert_page_table[index*2+1] & 0x3FF)
#define GET_PREV(vm,index) ((vm->invert_page_table[index*2+1]>>11) & 0x3FF)
#define GET_HEAD_NODE(vm) ((vm->invert_page_table[1]>>22) & 0x3FF)
#define GET_TAIL_NODE(vm) ((vm->invert_page_table[3]>>22) & 0x3FF)
#define SET_HEAD_NODE(vm,head) vm->invert_page_table[1] = ((vm->invert_page_table[1] & ~(0x3FF<<22))|((head&0x3FF)<<22))
#define SET_TAIL_NODE(vm,tail) vm->invert_page_table[1] = ((vm->invert_page_table[3] & ~(0x3FF<<22))|((tail&0x3FF)<<22))
#define IS_TAIL(vm,index) GET_BIT(10,vm->invert_page_table[index*2+1])
#define IS_HEAD(vm,index) GET_BIT(21,vm->invert_page_table[index*2+1])
#define IS_DIRTY(vm,index) GET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define IS_INVALID(vm,index) GET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])
#define SET_TAIL(vm,index) SET_BIT(10,vm->invert_page_table[index*2+1])
#define SET_HEAD(vm,index) SET_BIT(21,vm->invert_page_table[index*2+1])
#define SET_DIRTY(vm,index) SET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define SET_INVALID(vm,index) SET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])
#define SET_PID (vm,index,pid) vm->invert_page_table[index*2] = (vm->invert_page_table[index*2] & (~(0x3<<13)) | ((pid & 0x3) << 13))
#define UNSET_TAIL(vm,index) UNSET_BIT(10,vm->invert_page_table[index*2+1])
#define UNSET_HEAD(vm,index) UNSET_BIT(21,vm->invert_page_table[index*2+1])
#define UNSET_DIRTY(vm,index) UNSET_BIT(VALID_BIT,vm->invert_page_table[index*2])
#define UNSET_INVALID(vm,index) UNSET_BIT(DIRTY_BIT,vm->invert_page_table[index*2])
#define SET_ADDR(vm,index,addr) vm->invert_page_table[index*2] = (vm->invert_page_table[index*2]& ~(0x1FFF))|(a & 0x1FFF)
#define SET_NEXT(vm,index,next_index) vm->invert_page_table[index*2+1] = (vm->invert_page_table[index*2+1] & ~(0x3FF))|(next_index & 0x3FF)
#define SET_PREV(vm,index,prev_index) vm->invert_page_table[index*2+1] = (vm->invert_page_table[index*2+1] & ~(0x3FF<<11))| ((prev_index & 0x3FF)<<11)
#define INIT(vm,index) vm->invert_page_table[index*2] = 0; vm->invert_page_table[index*2+1] = 0
#define INIT_DISK_MAP(vm,index) vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] = \
(vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] & ~(0xFFFF<<((index%2)*16))) | (((1<<15))) << ((index%2)*16))
#define GET_DISK_IMAP(vm,index) ((vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2]) & (0xFFFF<<((index%2)*16))) >> ((index%2)*16)
#define GET_VM_FROM_DISK(vm,index) (GET_DISK_IMAP(vm,index) & 0x1FFF)
#define GET_PID_FROM_DISK(vm,index) ((GET_DISK_IMAP(vm,index) & 0x3<<13)>>13)
#define DISK_IS_INVALID(vm,index) (((GET_DISK_IMAP(vm,index)) & 0xFFFF) >> 15)
#define CONSTRUCT_DISK_IMAP(pid,vm_addr) ((((vm_addr & 0x1FFF) | ((pid & 0x3) <<13))) & ~(1<<15))
#define SET_DISK_TO_VM(vm,index,pid,vm_addr) vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] = \
(vm->invert_page_table[vm->PAGE_ENTRIES*2+index/2] & ~(0xFFFF<<((index%2)*16))) | (CONSTRUCT_DISK_IMAP(pid,vm_addr) << ((index%2)*16))
#define SET_COUNT (vm,count) vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] = ((vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] & ~(0x7FF<<15)) | (count<<15))
#define GET_COUNT (vm) (vm->invert_page_table[(vm->PAGE_ENTRIES-1)*2] & (0x7FF<<15)) >> 15 //11 bit count
__device__ u32 hash(u32 key, u32 i){
  return (key%1021+i)%1024;
}

__device__ u32 hash_disk(u32 key, u32 i){
  return (key%4099+i)%4096;
}

__device__ void init_invert_page_table(VirtualMemory *vm) {

  SET_COUNT(vm,0);
  SET_HEAD()
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    INIT(vm,i);
    SET_INVALID(vm,i);
  }

  for (int i=0; i < (vm->STORAGE_SIZE/vm->PAGE_SIZE);i++)
    INIT_DISK_MAP(vm,i);

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
    if(IS_TAIL(vm,GET_TAIL_NODE(vm)==0)){
      /* initial condition */
      SET_TAIL(vm,mem_entry);
      SET_TAIL_NODE(vm,mem_entry);
    }
    if(IS_HEAD(vm,GET_HEAD_NODE(vm)==0)){
      /* initial condition */
      SET_HEAD(vm,mem_entry);
      SET_HEAD_NODE(vm,mem_entry);
      return;
    }
    if(IS_TAIL(vm,mem_entry) && !IS_HEAD(vm,mem_entry)) {
      u32 new_tail = GET_PREV(vm,mem_entry);
      SET_TAIL(vm,new_tail);
      SET_TAIL_NODE(vm,new_tail);
      UNSET_TAIL(vm,mem_entry);
    }
    if(!IS_HEAD(vm,mem_entry)){
      u32 old_head = GET_HEAD_NODE(vm);
      UNSET_HEAD(vm,old_head);
      SET_HEAD(vm,mem_entry);
      SET_HEAD_NODE(vm,mem_entry);
      SET_NEXT(vm,mem_entry,old_head);
      SET_PREV(vm,old_head,mem_entry);
    }
}

__device__ void swap_in(VirtualMemory *vm,u32 mem_entry, u32 disk_entry) {
    for(u32 i=0;i<vm->PAGE_SIZE;i++)
       vm->buffer[(target<<5|(i&0x1F))] = vm->storage[(disk_entry<<5|(i&0x1F))];
    UNSET_DIRTY(vm,mem_entry); // fresh entry certainly not dirty
}

__device__ void swap_out(VirtualMemory *vm,u32 mem_entry, u32 disk_entry) {
    for(u32 i=0;i<vm->PAGE_SIZE;i++)
      vm->storage[(disk_entry<<5|(i&0x1F))] = vm->buffer[(target<<5|(i&0x1F))];
}

__device__ u32 extract_lru(VirtualMemory *vm) {
    u32 target = GET_TAIL_NODE(vm);
    UNSET_TAIL(vm,target);
    if(IS_HEAD(target)==0) {
      /** target not head, therefore has prev **/
      u32 new_tail = GET_PREV(vm,target);
      SET_TAIL(vm,new_tail);
      SET_TAIL_NODE(vm,new_tail);
    }
    return target;
}

__device__ u32 envict_lru(VirtualMemory *vm) {
    u32 target = extract_lru(vm);
    u32 disk_entry;
    for(int i=0; i < (vm->STORAGE_SIZE/vm->PAGE_SIZE); i++) {
      disk_entry = hash_disk(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break;
    }
    if(IS_DIRTY(vm,target)) swap_out(vm,target,disk_entry);
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
    for(int i=0; i < vm->PAGE_ENTRIES; i++) {
      mem_entry = hash(vm_page_addr,i);
      if(IS_INVALID(vm,mem_entry)) break;
      if(GET_ADDR(vm,mem_entry) == vm_page_addr) {
        mark_use(vm,mem_entry);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }

    }
    /** unable to find in mem, check disk **/
    for(int i=0; i < (vm->STORAGE_SIZE/vm->PAGE_SIZE); i++) {
      u32 disk_entry = hash_disk(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break; //cannot found in disk either, error
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr) {
        /* page allocation routine */
        UNSET_INVALID(vm,mem_entry);
        SET_ADDR(vm,mem_entry,vm_page_addr);
        SET_COUNT(GET_COUNT(vm)+1);
        /* swap_in and mark use */
        swap_in(vm,mem_entry,disk_entry);
        mark_use(vm,mem_entry);
        return m->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }

  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(IS_INVALID(vm,mem_entry)==0 && GET_ADDR(vm,mem_entry) == vm_page_addr){
        mark_use(vm,mem_entry);
        return vm->buffer[((mem_entry<<5) | vm_page_offset)];
      }
    }
    
    /** unable to find in mem, check disk **/
    for(int i=0; i < (vm->STORAGE_SIZE/vm->PAGE_SIZE); i++) {
      u32 disk_entry = hash_disk(vm_page_addr,i);
      if(DISK_IS_INVALID(vm,disk_entry)) break;
      if(GET_VM_FROM_DISK(vm,disk_entry)==vm_page_addr) {
        u32 destination = envict_lru(vm);
        /* page allocation routine */
        UNSET_INVALID(vm,destination);
        SET_ADDR(vm,destination,vm_page_addr);
        // SET_COUNT(GET_COUNT(vm)+1);
        // COUNT ALREADY MAX

        /* swap in and mark use */
        swap_in(vm,destination,disk_entry);
        mark_use(vm,destination);
        return m->buffer[((destination<<5) | vm_page_offset)];
      }
    }
  
  

  }
  return 0; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 vm_page_addr = (addr >> 5);
  u32 vm_page_offset = addr & 0x1F;
  if(GET_COUNT(vm) < vm->PAGE_ENTRIES) {
    /** non-full case, use hashing **/
    /** no need to consider replacement **/
    u32 mem_entry; // this can also mark the first empty slot appearing
    for(int i=0; i < vm->PAGE_ENTRIES; i++) {
      mem_entry = hash(vm_page_addr,i);
      if(IS_INVALID(vm,mem_entry)) break;
      if(GET_ADDR(vm,mem_entry) == vm_page_addr) {
        mark_use(vm,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        SET_DIRTY(vm,mem_entry);
        return;
      }
    }
    /** no record in mem, write directly to mem_entry **/

    /** page allocation routine **/
    UNSET_INVALID(vm,mem_entry);
    SET_COUNT(GET_COUNT(vm)+1);
    SET_DIRTY(vm,mem_entry); // dirty at born
    /** record new vm mapping **/
    SET_ADDR(vm,mem_entry,vm_page_addr);
    vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
    mark_use(vm,mem_entry);
  }
  else {
    /** full case, linear tranverse **/
    for(u32 mem_entry=0; mem_entry < vm->PAGE_ENTRIES; mem_entry++) {
      if(IS_INVALID(vm,mem_entry)==0 && GET_ADDR(vm,mem_entry) == vm_page_addr){
        SET_DIRTY(vm,mem_entry);
        mark_use(vm,mem_entry);
        vm->buffer[((mem_entry<<5) | vm_page_offset)] = value;
        return;
      }
    }
    
    /** unable to find in mem, envict a victim and place it there **/
    u32 destination = envict_lru(vm);
    UNSET_INVALID(vm,destination);
    /** record new vm mapping **/
    SET_ADDR(vm,destination,vm_page_addr);
    SET_DIRTY(vm,mem_entry);
    mark_use(vm,destination);
    vm->buffer[((destination<<5) | vm_page_offset)] = value;
    return;

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
}