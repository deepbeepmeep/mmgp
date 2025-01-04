from typing import Optional, Dict, List, Iterator, Tuple
from pathlib import Path
import torch
import mmap
import struct
import json
import base64
import safetensors
import accelerate
import os
from collections import OrderedDict


_old_torch_load_file = None
_old_safe_open = None



mmm = {}
verboseLevel = 1

import weakref

_map_to_dtype =  { 'BF16':  torch.bfloat16,  'U8': torch.uint8 , 'U16': torch.uint16, 'U32' : torch.uint32 , 'U64' : torch.uint64,
            'I8': torch.int8, 'I16': torch.int16, 'I32' : torch.int32 , 'I64' : torch.int64, 
            'F64' : torch.float64,  'F32': torch.float32, 'F16': torch.float16, 'BOOL' : torch.bool, "F8_E5M2" : torch.float8_e5m2, "F8_E4M3" : torch.float8_e4m3fn }


class MmapTracker:
    def __init__(self, file_path):
        self._maps = {}
        self._already_released = 0
        from pathlib import Path
        s = Path(file_path).parts
        if len(s)>2: 
            s = s[-2:]
        file_path = os.path.join(*s)
        self.file_path = file_path # os.path.abspath(file_path) 
        self.count = 0
        mmm[file_path] = self

    def register(self, mmap_obj, map_id, start, size):

        self.count += 1
        def finalizer(ref):
            self._already_released += 1
            if verboseLevel >=2:
                if self.count == self._already_released:
                    text =" (all the mmaps have been released)"
                else:
                    text =f" ({self.count-self._already_released:} left)"

                print(f"MMap Manager of file '{self.file_path}' : MMap no {map_id} has been released" + text)
            if self.count == self._already_released:
                del mmm[self.file_path]

            self._maps.pop(map_id, None)

        wr = weakref.ref(mmap_obj, finalizer)
        self._maps[map_id] = {
            'mmap' : wr,
            'start': start,
            'size': size,
            'end': start + size
        }
        return wr
       
    def get_active_maps(self):
        return dict(self._maps)


class cached_metadata:
    file_path = None
    file_length = 0
    file_date = None
    catalog = None
    metadata = None
    skip_bytes = 0

    def __init__(self, file_path, catalog, metadata, skip_bytes):
        self.catalog = catalog
        self.metadata = metadata
        self.skip_bytes = skip_bytes
        file_stats = os.stat(file_path)
        self.file_path = os.path.abspath(file_path)
        self.file_length = file_stats.st_size        
        self.file_date = file_stats.st_ctime

    def get_metadata(self, file_path):
        file_stats = os.stat(file_path)
        file_length = file_stats.st_size        
        file_date = file_stats.st_ctime
        file_path = os.path.abspath(file_path)
        if self.file_path != file_path or self.file_length != file_length or self.file_date != file_date:
            return None, None, None
        return self.catalog, self.metadata, self.skip_bytes
        
_cached_entry = None # ideally we should create a dict of the last n entries but one entry covers most cases

def  _parse_metadata(metadata):
    if metadata == None:
        return None
    
    new_metadata= {}
    
    for k,v in metadata.items():
        if k.endswith("_base64"):
            v_decoded = json.loads(base64.b64decode(v.encode('utf8')).decode('utf8'))
            p = k.rfind("_")
            new_k = k[:p]
            new_metadata[new_k]= v_decoded
        else:
            new_metadata[k] = v

    return new_metadata

def _read_safetensors_header(path, file):
    global _cached_entry
    length_of_header_bytes = file.read(8)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', length_of_header_bytes)[0]

    if _cached_entry != None:
        catalog, metadata, _ = _cached_entry.get_metadata(path)
    else:
        catalog = None

    if catalog == None:
        header_bytes = file.read(length_of_header)
        #catalog = json.loads(header_bytes.decode('utf-8'))
        catalog  = json.loads(header_bytes)
        metadata = catalog.pop("__metadata__", None) 
        metadata = _parse_metadata(metadata)

        _cached_entry = cached_metadata(path, catalog, metadata,length_of_header )        
    else:
        file.seek(length_of_header, 1)
    
    return catalog, metadata, length_of_header + 8

    
def torch_write_file(sd, file_path, quantization_map = None, config = None):
    from collections import OrderedDict
    sf_sd = OrderedDict()
 
    map = { torch.bfloat16 : 'BF16'  , torch.int64 : 'I64' , torch.int32 : 'I32' , torch.int16 : 'I16' , torch.int8 : 'I8' , 
           torch.uint64 : 'U64' , torch.uint32 : 'U32' , torch.uint16 : 'U16' , torch.uint8 : 'U8' , 
           torch.bool : 'BOOL' ,  torch.float64 : 'F64' , torch.float32 : 'F32' , torch.float16 : 'F16', torch.float8_e5m2 : "F8_E5M2", torch.float8_e4m3fn: "F8_E4M3" }
    pos = 0
    i = 0
    mx = 1000000
    for k , t  in sd.items():
        entry = {}
        dtypestr= map[t.dtype]
        entry["dtype"] = dtypestr  
        entry["shape"] = list(t.shape)
        size = torch.numel(t) * t.element_size()
        entry["data_offsets"] = [pos, pos + size]
        pos += size
        sf_sd[k] = entry
        i+=1
        if i==mx:
            break
    metadata = dict()
    if not quantization_map is None:
        metadata["quantization_format"] = "quanto"
        metadata["quantization_map_base64"] =  base64.b64encode(json.dumps(quantization_map, ensure_ascii=False).encode('utf8')).decode('utf8')  

    if not config is None:
        metadata["config_base64"] = base64.b64encode(json.dumps(config, ensure_ascii=False).encode('utf8')).decode('utf8')

    if len(metadata) > 0:
        sf_sd["__metadata__"] = metadata

    header_bytes = json.dumps(sf_sd).encode()
    #header_bytes =json.dumps(config, ensure_ascii=False).encode('utf8')    
    size_header = len(header_bytes)
    import struct

    length_of_header_bytes = struct.pack('<Q', size_header)

    empty_tensor = b'\x80\x3f'

    with open(file_path, "wb") as writer:
        bytes_written = writer.write(length_of_header_bytes)        
        bytes_written = writer.write(header_bytes)        

        i = 0
        for k , t  in sd.items():
            size = torch.numel(t) * t.element_size()
            if len(t.shape) == 0:
                bytes_written = writer.write(empty_tensor)
            else:
                buffer = t.view(torch.uint8).numpy().tobytes()
                bytes_written = writer.write(buffer)
            assert bytes_written == size
            i+=1
            if i==mx:
                break

class SafeTensorFile:
    """Main class for accessing safetensors files that provides memory-efficient access"""
    
    def __init__(self, file_path, metadata, catalog, skip_bytes):
        self._file_path = file_path
        self._metadata = metadata
        self._catalog = catalog
        self._skip_bytes = skip_bytes
        self._keys = None
        self.sd = None
        self.mtracker = None

    @classmethod
    def load_metadata(cls, file_path):    
        with open(file_path, 'rb') as f:
            catalog, metadata, skip_bytes = _read_safetensors_header(file_path, f)

        return cls(file_path, metadata, catalog, skip_bytes)

    def init_tensors(self):
        if self.sd is None:
            self.sd = self.create_tensors()
        return self.sd
    
    def create_tensors(self):
 
        self.mtracker = MmapTracker(self._file_path)
        import mmap

        PAGE_SIZE =  mmap.ALLOCATIONGRANULARITY 
        MMAP_SIZE = 1024 * 1024 * 1024  # 1GB

        # First pass: find optimal aligned map boundaries
        skip_bytes = self._skip_bytes
        tensor_map_indexes  = []
        maps_info = []
        current_pos = skip_bytes
        current_map_start = (skip_bytes // PAGE_SIZE) * PAGE_SIZE
        current_map_size = skip_bytes - current_map_start
        idx = 0
        for k,v in self._catalog.items():
            data_offsets = v["data_offsets"]
            length = data_offsets[1]-data_offsets[0]
            if current_map_size + length > MMAP_SIZE:
                maps_info.append((current_map_start, current_map_size))
                current_map_start = (current_pos // PAGE_SIZE) * PAGE_SIZE
                current_map_size = current_pos - current_map_start
                idx += 1
            tensor_map_indexes.append(idx)
            current_map_size += length
            current_pos += length
    
        maps_info.append((current_map_start, current_map_size))
        
        # Second pass: create maps and tensors
        maps = []
        sd = OrderedDict()    
        
        current_pos = skip_bytes
        with open(self._file_path, 'rb') as f:
            i = 0
            for map_start, map_size in maps_info:
                mm = mmap.mmap(f.fileno(), map_size, offset=map_start, access=mmap.ACCESS_COPY) #.ACCESS_READ
                maps.append((mm, map_start, map_size))
                self.mtracker.register(mm, i, map_start, map_size)
                i = i+ 1

            iter_tensor_no = iter(tensor_map_indexes)
            for k,v in self._catalog.items():
                dtypestr =  v["dtype"]
                dtype= _map_to_dtype[dtypestr]
                shape = v["shape"]
                data_offsets = v["data_offsets"]
                length = data_offsets[1]-data_offsets[0]
                map_idx = next(iter_tensor_no)
                offset = current_pos - maps[map_idx][1]
                if len(shape) == 0:
                    t = torch.ones((), dtype=dtype, device="cpu")
                else:
                    mv = memoryview(maps[map_idx][0])[offset:offset + length]                
                    t = torch.frombuffer(mv, dtype=dtype)
                    t = torch.reshape(t, shape)
                # t._mmap = maps[map_idx][0]
                sd[k] = t
                current_pos += length

        return sd

    def get_tensor(self, name: str) -> torch.tensor:
        """Get a tensor by name"""
        self.init_tensors()
        return self.sd[name]
 
    def keys(self) -> List[str]:
        """Get list of tensor names"""
        if self._keys is None:
            self._keys = list(self._catalog)
        return self._keys
        
    def names(self) -> List[str]:
        """Alias for keys()"""
        return self.keys()
        
    def tensors(self) -> Dict[str, torch.tensor]:
        """Get dictionary of all tensors"""
        self.init_tensors()
        return self.sd
        
    def metadata(self) -> Optional[Dict[str, str]]:
        """Get metadata dictionary"""
        return self._metadata
        
    def __len__(self) -> int:
        """Get number of tensors"""
        self.init_tensors()
        return len(self.keys())
        
    def __contains__(self, key: str) -> bool:
        """Check if tensor exists"""
        return key in self.keys()
        
    def __iter__(self) -> Iterator[Tuple[str, torch.tensor ]]:
        """Iterate over (name, tensor) pairs"""
        return ((name, self.get_tensor(name)) for name in self.keys())

    def _free_resources(self):
        del self.sd
        del self._catalog 
        
class _SafeTensorLoader:
    """Context manager for loading SafeTensorFile"""
    
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.sft = None
        
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
            
    def __enter__(self) -> SafeTensorFile:
        """Open file and return SafeTensorFile instance"""
        
        try:
            self.sft = SafeTensorFile.load_metadata(self.filename)
            return self.sft 
            
        except Exception as e:
            self.close()
            raise Exception(f"Failed to load safetensors file: {e}") from e
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources"""        
        self.close()
        
    def close(self) -> None:
        if self.sft != None:
            self.sft._free_resources()
        pass


def safe_open(filename: str, framework: str = "pt",device = "cpu") -> _SafeTensorLoader:
    if device != "cpu" or framework !="pt":
        pass
        return _old_safe_open(filename =filename, framework=framework, device=device)
    return _SafeTensorLoader(filename)

def torch_load_file( filename, device = 'cpu' ) -> Dict[str, torch.Tensor]:
    sd = {}
    with safe_open(filename, framework="pt", device = device ) as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
        return sd

_old_torch_load_file = safetensors.torch.load_file  
safetensors.torch.load_file = torch_load_file
_old_safe_open = safetensors.safe_open
safetensors.safe_open = safe_open
accelerate.utils.modeling.safe_open = safe_open
accelerate.utils.modeling.safe_load_file = torch_load_file
try:
    import transformers
    transformers.modeling_utils.safe_open = safe_open
    transformers.modeling_utils.safe_load_file = torch_load_file
except:
    pass

