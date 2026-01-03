# ------------------ Memory Management 3.6.10 for the GPU Poor by DeepBeepMeep (mmgp)------------------
#
# This module contains multiples optimisations so that models such as Flux (and derived), Mochi, CogView, HunyuanVideo, ...  can run smoothly on a 24 GB GPU limited card. 
# This a replacement for the accelerate library that should in theory manage offloading, but doesn't work properly with models that are loaded / unloaded several
# times in a pipe (eg VAE).
#
# Requirements:
# - VRAM: minimum 12 GB, recommended 24 GB (RTX 3090/ RTX 4090)
# - RAM: minimum 24 GB, recommended 48 - 64 GB 
# 
# It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
# Make sure that the pipeline explictly loads the models in the CPU device 
#   for instance: pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
# For a quick setup, you may want to choose between 5 profiles depending on your hardware, for instance:
#   from mmgp import offload, profile_type
#   offload.profile(pipe, profile_type.HighRAM_LowVRAM_Fast)
# Alternatively you may want to your own parameters, for instance:
#   from mmgp import offload
#   offload.all(pipe, pinToMemory=true, extraModelsToQuantize = ["text_encoder_2"] )
# The 'transformer' model that contains usually the video or image generator is quantized on the fly by default to 8 bits so that it can fit into 24 GB of VRAM. 
# You can prevent the transformer quantization by adding the parameter quantizeTransformer = False
# If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. In that case you need to set the option quantizeTransformer to False to turn off on the fly quantization.
# You can specify a list of additional models string ids to quantize (for instance the text_encoder) using the optional argument extraModelsToQuantize. This may be useful if you have less than 48 GB of RAM.
# Note that there is little advantage on the GPU / VRAM side to quantize text encoders as their inputs are usually quite light. 
# Conversely if you have more than 48GB RAM you may want to enable RAM pinning with the option pinnedMemory = True. You will get in return super fast loading / unloading of models
# (this can save significant time if the same pipeline is run multiple times in a row)
# 
# Sometime there isn't an explicit pipe object as each submodel is loaded separately in the main app. If this is the case, you need to create a dictionary that manually maps all the models.
#
# For instance :
# for flux derived models: pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
# for mochi: pipe = { "text_encoder": self.text_encoder, "transformer": self.dit, "vae":self.decoder }
#
# Please note that there should be always one model whose Id is 'transformer'. It corresponds to the main image / video model which usually needs to be quantized (this is done on the fly by default when loading the model)
# 
# Becareful, lots of models use the T5 XXL as a text encoder. However, quite often their corresponding pipeline configurations point at the official Google T5 XXL repository 
# where there is a huge 40GB model to download and load. It is cumbersorme as it is a 32 bits model and contains the decoder part of T5 that is not used. 
# I suggest you use instead one of the 16 bits encoder only version available around, for instance:
# text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=torch.float16)
#
# Sometime just providing the pipe won't be sufficient as you will need to change the content of the core model: 
# - For instance you may need to disable an existing CPU offload logic that already exists (such as manual calls to move tensors between cuda and the cpu)
# - mmpg to tries to fake the device as being "cuda" but sometimes some code won't be fooled and it will create tensors in the cpu device and this may cause some issues.
# 
# You are free to use my module for non commercial use as long you give me proper credits. You may contact me on twitter @deepbeepmeep
#
# Thanks to
# ---------
# Huggingface / accelerate for the hooking examples
# Huggingface / quanto for their very useful quantizer
# gau-nernst for his Pinnig RAM samples


#

import torch
import gc
import time
import functools
import sys
import os
import json
import inspect
import psutil
import builtins
from accelerate import init_empty_weights
from functools import wraps
import functools
import types

from mmgp import safetensors2
from mmgp import profile_type
from .quant_router import detect_safetensors_format, detect_and_convert, apply_pre_quantization
from optimum.quanto import freeze,  qfloat8, qint4 , qint8, quantize, QModuleMixin, QLinear, QTensor,  quantize_module, register_qmodule

# support for Embedding module quantization that is not supported by default by quanto
@register_qmodule(torch.nn.Embedding)
class QEmbedding(QModuleMixin, torch.nn.Embedding):
    bias = None
    @classmethod
    def qcreate(cls, module, weights, activations = None, optimizer = None, device = None):
        module.bias = None
        return cls( module.num_embeddings, module.embedding_dim, module.padding_idx , module.max_norm, module.norm_type, module.scale_grad_by_freq, module.sparse, dtype=module.weight.dtype, device=device, weights=weights,
                    activations=activations, optimizer=optimizer, quantize_input=True)      
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding( input, self.qweight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse )



def cudacontext(device):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with torch.device(device):
                return func(*args, **kwargs)
        return wrapper
    return decorator


shared_state = {}

def get_cache(cache_name):
    all_cache = shared_state.get("_cache",  None)
    if all_cache is None:
        all_cache = {}
        shared_state["_cache"]=  all_cache
    cache = all_cache.get(cache_name, None)
    if cache is None:
        cache = {}
        all_cache[cache_name] = cache
    return cache

def clear_caches():
    all_cache = shared_state.get("_cache",  None)
    if all_cache is not None:
        all_cache.clear()


def get_available_qtypes():
    try:
        from optimum.quanto.tensor.qtype import qtypes as _quanto_qtypes
    except Exception:
        return []
    return sorted(_quanto_qtypes.keys())


def get_available_qtype_aliases():
    aliases = set()
    for name in get_available_qtypes():
        key = str(name).lower()
        aliases.add(key)
        if key.startswith("q") and len(key) > 1:
            aliases.add(key[1:])
        if "float8" in key:
            aliases.add("fp8")
    return aliases


def get_quantization_tokens(quantization):
    if quantization is None:
        return []
    key = str(quantization).lower()
    if len(key) == 0:
        return []
    aliases = get_available_qtype_aliases()
    if key not in aliases:
        return []
    tokens = {key}
    if key.startswith("q") and len(key) > 1:
        tokens.add(key[1:])
    if "float8" in key or key == "fp8":
        tokens.add("fp8")
    if "int4" in key:
        tokens.add("int4")
    if "int8" in key:
        tokens.add("int8")
    return sorted(tokens, key=len, reverse=True)


def get_quantization_label(quantization):
    if quantization is None:
        return ""
    key = str(quantization).lower()
    if key in ("", "none", "bf16", "fp16", "float16", "bfloat16"):
        return ""
    aliases = get_available_qtype_aliases()
    if key not in aliases:
        return ""
    if "float8" in key or key == "fp8":
        return "FP8"
    if key.startswith("q"):
        key = key[1:]
    return key.replace("_", " ").upper()


_quantization_filename_cache = {}


def _normalize_quant_file_key(file_path):
    try:
        return os.path.normcase(os.path.abspath(file_path))
    except Exception:
        return str(file_path).lower()


def get_cached_quantization_for_file(file_path):
    if not file_path:
        return None
    return _quantization_filename_cache.get(_normalize_quant_file_key(file_path))


def cache_quantization_for_file(file_path, kind):
    if not file_path or not kind:
        return
    key = _normalize_quant_file_key(file_path)
    if key not in _quantization_filename_cache:
        _quantization_filename_cache[key] = kind


def _infer_qtype_from_quantization_map(quantization_map):
    if not quantization_map:
        return None
    counts = {}
    for entry in quantization_map.values():
        if not isinstance(entry, dict):
            continue
        weights = entry.get("weights")
        if not weights or weights == "none":
            continue
        counts[weights] = counts.get(weights, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_quantization_kind_for_file(file_path, verboseLevel=1):
    cached = get_cached_quantization_for_file(file_path)
    if cached:
        return cached
    if not file_path or not os.path.isfile(file_path):
        return None
    if not (".safetensors" in file_path or ".sft" in file_path):
        return None
    try:
        state_dict, metadata = _safetensors_load_file(file_path, writable_tensors=False)
    except Exception:
        return None
    kind = None
    try:
        info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel)
        kind = info.get("kind")
    except Exception:
        kind = None
    if (not kind or kind == "none") and metadata is not None:
        inferred = _infer_qtype_from_quantization_map(metadata.get("quantization_map"))
        if inferred:
            kind = inferred
    cache_quantization_for_file(file_path, kind or "none")
    return kind


def detect_quantization_label_from_filename(filename):
    if not filename:
        return ""
    cached = get_cached_quantization_for_file(filename)
    if cached:
        return get_quantization_label(cached)
    kind = detect_quantization_kind_for_file(filename, verboseLevel=0)
    if kind:
        label = get_quantization_label(kind)
        if label:
            return label
    base = os.path.basename(filename).lower()
    for token in sorted(get_available_qtype_aliases(), key=len, reverse=True):
        if token and token in base:
            return get_quantization_label(token)
    if "quanto" in base:
        return "QUANTO"
    return ""


mmm = safetensors2.mmm

default_verboseLevel = 1

ONE_MB =  1048576
sizeofhalffloat = torch.bfloat16.itemsize
sizeofint8 = torch.int8.itemsize
total_pinned_bytes = 0
max_pinnable_bytes = 0

physical_memory= psutil.virtual_memory().total

HEADER = '\033[95m'
ENDC = '\033[0m'
BOLD ='\033[1m'
UNBOLD ='\033[0m'

class clock:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    @classmethod
    def start(cls):
        self = cls()        
        self.start_time =time.time()
        return self        

    def stop(self):
        self.stop_time =time.time()  

    def time_gap(self):
        return self.stop_time - self.start_time
    
    def format_time_gap(self):
        return f"{self.stop_time - self.start_time:.2f}s"

# useful functions to move a group of tensors (to design custom offload patches)
def move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        _dict = {}
        for k, v in obj.items():
            _dict[k] = move_tensors(v, device)
        return _dict
    elif isinstance(obj, list):
        _list = []
        for v in obj:
            _list.append(move_tensors(v, device))
        return _list
    else:
        raise TypeError("Tensor or list / dict of tensors expected")
def _get_module_name(v):
    return v.__module__.lower()


def _compute_verbose_level(level):
    if level <0:        
        level = safetensors2.verboseLevel = default_verboseLevel
    safetensors2.verboseLevel = level
    return level

def _get_perc_reserved_mem_max(perc_reserved_mem_max = 0):
    if perc_reserved_mem_max <=0:
        perc_reserved_mem_max = os.getenv("perc_reserved_mem_max", 0)

    if perc_reserved_mem_max <= 0:             
        perc_reserved_mem_max = 0.40 if os.name == 'nt' else 0.5        
    return perc_reserved_mem_max
    
def _get_max_reservable_memory(perc_reserved_mem_max = 0):
    max_reservable_memory = perc_reserved_mem_max * physical_memory
    return  max_reservable_memory

def _detect_main_towers(model, min_floors = 5):
    cur_blocks_prefix = None
    towers_modules= []
    towers_names= []

    floors_modules= []
    tower_name = None


    for submodule_name, submodule in model.named_modules():  

        if submodule_name=='':
            continue

        if cur_blocks_prefix != None:
            if submodule_name.startswith(cur_blocks_prefix):
                depth_prefix = cur_blocks_prefix.split(".")
                depth_name = submodule_name.split(".")
                level  =  depth_name[len(depth_prefix)-1]                        
                pre , num = _extract_num_from_str(level)

                if num != cur_blocks_seq: 
                    floors_modules.append(submodule)

                cur_blocks_seq = num
            else:
                if len(floors_modules) >= min_floors:
                    towers_modules += floors_modules
                    towers_names.append(tower_name)
                tower_name = None
                floors_modules= []
                cur_blocks_prefix, cur_blocks_seq = None, -1

        if cur_blocks_prefix == None:
            pre , num = _extract_num_from_str(submodule_name)
            if isinstance(submodule, (torch.nn.ModuleList)):  
                cur_blocks_prefix, cur_blocks_seq = pre + ".",  -1
                tower_name = submodule_name + "." 
            elif num >=0:
                cur_blocks_prefix, cur_blocks_seq = pre, num
                tower_name = submodule_name[ :-1]  
                floors_modules.append(submodule)

    if len(floors_modules) >= min_floors:
        towers_modules += floors_modules
        towers_names.append(tower_name)

    return towers_names, towers_modules



def _get_model(model_path):
    if os.path.isfile(model_path):
        return model_path
    
    from pathlib import Path
    _path = Path(model_path).parts
    _filename = _path[-1]
    _path = _path[:-1]
    if len(_path)<=1:
        raise Exception("file not found")
    else:
        try:
            from huggingface_hub import  hf_hub_download #snapshot_download,    
            repoId=  os.path.join(*_path[0:2] ).replace("\\", "/")

            if len(_path) > 2:
                _subfolder = os.path.join(*_path[2:] )
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename,  subfolder=_subfolder)
            else:
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename)
        except:
           model_path = None 
    return model_path



def _remove_model_wrapper(model):
    if not model._modules is None:
        if len(model._modules)!=1:
            return model
    sub_module = model._modules[next(iter(model._modules))]
    if hasattr(sub_module,"config") or hasattr(sub_module,"base_model"):
        return sub_module
    return model  

 

def _move_to_pinned_tensor(source_tensor, big_tensor, offset, length):
    dtype= source_tensor.dtype
    shape = source_tensor.shape
    if len(shape) > 0 :
        t = source_tensor.view(torch.uint8)
        t = torch.reshape(t, (length,))
    else:
        t = source_tensor
    # magic swap !
    big_tensor[offset: offset + length] = t 
    t = big_tensor[offset: offset + length]
    t = t.view(dtype)
    t = torch.reshape(t, shape)
    assert t.is_pinned()
    return t

def _safetensors_load_file(file_path, writable_tensors = True):
    from collections import OrderedDict
    sd = OrderedDict()    

    with safetensors2.safe_open(file_path, framework="pt", device="cpu", writable_tensors =writable_tensors) as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
        metadata = f.metadata()

    return sd, metadata

def _force_load_buffer(p):
    # To do : check if buffer was persistent and transfer state, or maybe swap keep already this property ?
    q = torch.nn.Buffer(p.clone())
    torch.utils.swap_tensors(p, q)
    del q

def _force_load_parameter(p):
    q = torch.nn.Parameter(p.clone())
    torch.utils.swap_tensors(p, q)
    del q

def _unwrap_quantized_tensor(tensor):
    if hasattr(tensor, "_data") and torch.is_tensor(tensor._data):
        return tensor._data
    return tensor

def _qtensor_get_quantized_subtensors(self):
    subtensors = []
    if getattr(self, "_qtype", None) == qint4:
        data = _unwrap_quantized_tensor(self._data)
        subtensors.append(("data", data))
        if hasattr(self, "_scale_shift") and self._scale_shift is not None:
            subtensors.append(("scale_shift", self._scale_shift))
        else:
            if hasattr(self, "_scale") and self._scale is not None:
                subtensors.append(("scale", self._scale))
            if hasattr(self, "_shift") and self._shift is not None:
                subtensors.append(("shift", self._shift))
        return subtensors

    if hasattr(self, "_data"):
        data = _unwrap_quantized_tensor(self._data)
        subtensors.append(("data", data))
    if hasattr(self, "_scale") and self._scale is not None:
        subtensors.append(("scale", self._scale))
    return subtensors

def _qtensor_set_quantized_subtensors(self, sub_tensors):
    if isinstance(sub_tensors, dict):
        sub_map = sub_tensors
    else:
        sub_map = {name: tensor for name, tensor in sub_tensors}

    data = sub_map.get("data", None)
    if data is not None:
        if hasattr(self, "_data") and hasattr(self._data, "_data") and torch.is_tensor(self._data._data):
            self._data._data = data
        else:
            self._data = data

    if getattr(self, "_qtype", None) == qint4:
        if "scale_shift" in sub_map and sub_map["scale_shift"] is not None:
            self._scale_shift = sub_map["scale_shift"]
        else:
            if "scale" in sub_map and sub_map["scale"] is not None:
                self._scale = sub_map["scale"]
            if "shift" in sub_map and sub_map["shift"] is not None:
                self._shift = sub_map["shift"]
    else:
        if "scale" in sub_map and sub_map["scale"] is not None:
            self._scale = sub_map["scale"]

if not hasattr(QTensor, "get_quantized_subtensors"):
    QTensor.get_quantized_subtensors = _qtensor_get_quantized_subtensors
if not hasattr(QTensor, "set_quantized_subtensors"):
    QTensor.set_quantized_subtensors = _qtensor_set_quantized_subtensors

def _get_quantized_subtensors(p):
    getter = getattr(p, "get_quantized_subtensors", None)
    if getter is None:
        return None
    sub_tensors = getter()
    if not sub_tensors:
        return None
    if isinstance(sub_tensors, dict):
        sub_tensors = list(sub_tensors.items())
    out = []
    for name, tensor in sub_tensors:
        if tensor is None:
            continue
        if torch.is_tensor(tensor):
            out.append((name, tensor))
    return out if out else None

def _set_quantized_subtensors(p, sub_tensors):
    setter = getattr(p, "set_quantized_subtensors", None)
    if setter is None:
        return False
    setter(sub_tensors)
    return True

def _subtensors_nbytes(sub_tensors):
    return sum(torch.numel(t) * t.element_size() for _, t in sub_tensors)

def _subtensors_itemsize(sub_tensors, fallback):
    for _, t in sub_tensors:
        return t.element_size()
    return fallback

def _get_tensor_ref(p):
    sub_tensors = _get_quantized_subtensors(p)
    if sub_tensors:
        for _, t in sub_tensors:
            ref = t.data_ptr()
            del sub_tensors
            return ref
        del sub_tensors
    return p.data_ptr()


BIG_TENSOR_MAX_SIZE = 2**28 # 256 MB
BIG_TENSOR_MIN_SIZE = 2**26 # 64 MB
RESERVED_RAM_MIN_AVAILABLE = BIG_TENSOR_MAX_SIZE # 2**27 # 128 MB

def _extract_tie_weights_from_sd(sd , sd_name, verboseLevel =1):
    tied_weights = {}
    tied_weights_count = 0
    tied_weights_total = 0
    tied_weights_last = None
    ref_cache = {}

    for n, p in sd.items():
        ref = _get_tensor_ref(p)
        match = ref_cache.get(ref, None)
        if match != None:
            match_name, match_size = match
            tied_weights_count += 1
            tied_weights_total += match_size
            if verboseLevel >=1:
                tied_weights_last = f"{match_name} <-> {n}"
            tied_weights[n] = match_name
        else:
            length = torch.numel(p.data) * p.data.element_size() 
            ref_cache[ref] = (n, length)
        
    if verboseLevel >=1 and tied_weights_count > 0:
        if  tied_weights_count == 1:
            print(f"Tied weights of {tied_weights_total/ONE_MB:0.2f} MB detected: {tied_weights_last}")
        else:
            print(f"Found {tied_weights_count} tied weights for a total of {tied_weights_total/ONE_MB:0.2f} MB, last : {tied_weights_last}")

def _pin_sd_to_memory(sd, sd_name, tied_weights = None, gig_tensor_size = BIG_TENSOR_MAX_SIZE, verboseLevel = 1):
    global max_pinnable_bytes, total_pinned_bytes


    names_list = sd_name if isinstance(sd, list) else [sd_name]

    if max_pinnable_bytes > 0 and  total_pinned_bytes >= max_pinnable_bytes:
        if  verboseLevel>=1 :
            print(f"Unable to pin data of '{','.join(names_list)}' to reserved RAM as there is no reserved RAM left. Transfer speed from RAM to VRAM will may be slower.")
        return

    
    if isinstance(sd, list):
        new_sd = {}
        for i, sub_sd,  in enumerate(sd):
            for k, v in sub_sd.items():
                new_sd[str(i) + "#" + k] =v
        sd = new_sd
        del new_sd
        sub_sd = None

    if isinstance(tied_weights, list):
        new_tied_weights = {}
        for i, sub_tied_weights,  in enumerate(tied_weights):
            for k, v in sub_tied_weights.items():
                new_tied_weights[str(i) + "#" + k] =v
        sd = new_tied_weights
        del new_tied_weights
        sub_tied_weights = None

    current_big_tensor_size = 0
    big_tensor_no  = 0
    big_tensors_sizes = []
    tensor_map_indexes = []
    total_tensor_bytes = 0

    for n, p in sd.items():
        if tied_weights == None or not n in tied_weights :
            length = torch.numel(p.data) * p.data.element_size() 

            if current_big_tensor_size + length > gig_tensor_size :
                big_tensors_sizes.append(current_big_tensor_size)
                current_big_tensor_size = 0
                big_tensor_no += 1

            itemsize = p.data.dtype.itemsize
            if current_big_tensor_size % itemsize:
                current_big_tensor_size += itemsize - current_big_tensor_size % itemsize
            tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length  ))
            current_big_tensor_size += length

            total_tensor_bytes += length
  
    big_tensors_sizes.append(current_big_tensor_size)

    big_tensors = []
    last_big_tensor = 0
    total = 0  
    incomplete_pinning = False

    try:
        dummy_pinned_tensor = torch.empty( RESERVED_RAM_MIN_AVAILABLE, dtype= torch.uint8, pin_memory=True, device="cpu")
    except:
        print("There isn't any Reserved RAM left, you may need to choose a profile with a higher number that requires less Reserved RAM or set OS env 'perc_reserved_mem_max' to a value less 0.3")
        gc.collect()
        torch.cuda.empty_cache()
        return
    
    for size in big_tensors_sizes:
        try:
            current_big_tensor = torch.empty( size, dtype= torch.uint8, pin_memory=True, device="cpu")
            big_tensors.append(current_big_tensor)
        except:
            incomplete_pinning = True
            print(f"Unable to pin more tensors for '{sd_name}' as the maximum reservable memory has been reached ({total/ONE_MB:.2f}). Transfer speed from RAM to VRAM may be slower.")
            break

        last_big_tensor += 1
        total += size
    del dummy_pinned_tensor

        
    tensor_no = 0
    # prev_big_tensor = 0
    q_name = None
    for n, p  in sd.items():
        if tied_weights != None:
            q_name = tied_weights.get(n,None)
        if q_name != None:
            q = sd[q_name] 
            p.data = q.data
            assert p.data.is_pinned()
            q = None
        else:
            big_tensor_no, offset, length = tensor_map_indexes[tensor_no]
 
            if big_tensor_no>=0 and big_tensor_no < last_big_tensor:
                current_big_tensor = big_tensors[big_tensor_no]
                length = torch.numel(p.data) * p.data.element_size()
                q = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length)
                torch.utils.swap_tensors(p, q)
                del q 
            tensor_no += 1
        del p
    # global total_pinned_bytes
    # total_pinned_bytes += total
    gc.collect()
    torch.cuda.empty_cache()


    if verboseLevel >=1:
        if incomplete_pinning :
            if len(names_list) > 1:
                print(f"'{','.join(names_list)}' were partially pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
            else:
                print(f"'{','.join(names_list)}' was partially pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
        else:
            if len(names_list) > 1:
                print(f"'{','.join(names_list)}' were pinned entirely to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
            else:
                print(f"'{','.join(names_list)}' was pinned entirely to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")


    return 


def _pin_to_memory(model, model_id, partialPinning = False, pinnedPEFTLora = True, big_tensor_size = BIG_TENSOR_MAX_SIZE, perc_reserved_mem_max = 0,verboseLevel = 1):

    global max_pinnable_bytes, total_pinned_bytes
    if max_pinnable_bytes > 0 and  total_pinned_bytes >= max_pinnable_bytes:

        if  verboseLevel>=1 :
            print(f"Unable to pin data of '{model_id}' to reserved RAM as there is no reserved RAM left. Transfer speed from RAM to VRAM may be slower.")
        return
    
    if partialPinning:
        towers_names, _ = _detect_main_towers(model)


    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max)
    max_reservable_memory = _get_max_reservable_memory(perc_reserved_mem_max) 

    current_big_tensor_size = 0
    big_tensor_no  = 0
    big_tensors_sizes = []
    tensor_map_indexes = []
    total_tensor_bytes = 0

    params_dict = {} #  OrderedDict
    for k, sub_module in model.named_modules():
        include = True
        if partialPinning:
            include = any(k.startswith(pre) for pre in towers_names) if partialPinning else True
        if include and not pinnedPEFTLora and ".lora_" in k:
            include = False

        if include:
            params_dict.update( { k + '.' + n : (p,  False) for n, p in sub_module.named_parameters(recurse=False) }  )
            params_dict.update( { k + '.' + n : (b,  True) for n, b in sub_module.named_buffers(recurse=False) }  )

    if  verboseLevel>=1 :
        if partialPinning:
            if len(params_dict) == 0:
                print(f"Unable to apply Partial of '{model_id}' as no isolated main structures were found")
            else:
                print(f"Partial pinning of data of '{model_id}' to reserved RAM")
        else:            
            print(f"Pinning data of '{model_id}' to reserved RAM")

    if len(params_dict) == 0:
        return

    ref_cache = {}
    tied_weights = {}
    tied_weights_count = 0
    tied_weights_total = 0
    tied_weights_last = None

    for n, (p, _) in params_dict.items():
        ref = _get_tensor_ref(p)
        match = ref_cache.get(ref, None)
        if match != None:
            match_name, match_size = match
            tied_weights_count += 1
            tied_weights_total += match_size
            if verboseLevel >=1:
                tied_weights_last = f"{match_name} <-> {n}"
            tied_weights[n] = match_name
        else:
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                if builtins.all(t.is_pinned() for _, t in sub_tensors):
                    params_dict[n] = (None, False)
                    del sub_tensors
                    continue
                length = _subtensors_nbytes(sub_tensors)
            else:
                if p.data.is_pinned():
                    params_dict[n] = (None, False)
                    continue
                length = torch.numel(p.data) * p.data.element_size()

            ref_cache[ref] = (n, length)
            if current_big_tensor_size + length > big_tensor_size and current_big_tensor_size !=0  :
                big_tensors_sizes.append(current_big_tensor_size)
                current_big_tensor_size = 0
                big_tensor_no += 1

            if sub_tensors:
                itemsize = _subtensors_itemsize(sub_tensors, p.data.dtype.itemsize)
                del sub_tensors
            else:
                itemsize = p.data.dtype.itemsize
            if current_big_tensor_size % itemsize:
                current_big_tensor_size += itemsize - current_big_tensor_size % itemsize
            tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length  ))
            current_big_tensor_size += length

            total_tensor_bytes += length
    p = None
    if verboseLevel >=1 and tied_weights_count > 0:
        if  tied_weights_count == 1:
            print(f"Tied weights of {tied_weights_total/ONE_MB:0.2f} MB detected: {tied_weights_last}")
        else:
            print(f"Found {tied_weights_count} tied weights for a total of {tied_weights_total/ONE_MB:0.2f} MB, last : {tied_weights_last}")
                

    big_tensors_sizes.append(current_big_tensor_size)

    big_tensors = []
    total = 0
    

    failed_planned_allocation = False
    gc.collect()
    try:
        dummy_pinned_tensor = torch.empty( RESERVED_RAM_MIN_AVAILABLE, dtype= torch.uint8, pin_memory=True, device="cpu")
    except:
        print("There isn't any Reserved RAM left, you may need to choose a profile with a higher number that requires less Reserved RAM or set OS env 'perc_reserved_mem_max' to a value less than{perc_reserved_mem_max}")
        return

    last_allocated_big_tensor = -1        
    tensor_no = 0
    # prev_big_tensor = 0
    for n, (p, is_buffer) in params_dict.items():
        if p is None: continue
        q_name = tied_weights.get(n,None)
        if q_name != None:
            q , _ = params_dict[q_name] 
            sub_tensors = _get_quantized_subtensors(q)
            if sub_tensors:
                sub_map = {name: tensor for name, tensor in sub_tensors}
                _set_quantized_subtensors(p, sub_map)
                del sub_map, sub_tensors
            else:
                p.data = q.data
                assert p.data.is_pinned()
            q = None
        else:

            big_tensor_no, offset, length = tensor_map_indexes[tensor_no]
            if last_allocated_big_tensor <  big_tensor_no:
                last_allocated_big_tensor += 1
                size = max(big_tensors_sizes[last_allocated_big_tensor], BIG_TENSOR_MIN_SIZE) 
                try:
                    if max_reservable_memory > 0 and ( (total_pinned_bytes + total + size) >= max_reservable_memory):
                        dummy_pinned_tensor = None
                        failed_planned_allocation = True
                        max_pinnable_bytes = total_pinned_bytes + total
                        break

                    current_big_tensor = torch.empty( size, dtype= torch.uint8, pin_memory=True, device="cpu")
                    big_tensors.append(current_big_tensor)
                except:
                    print(f"Unable to pin more tensors for this model as the maximum reservable memory has been reached ({total/ONE_MB:.2f}).")
                    dummy_pinned_tensor = None
                    failed_planned_allocation = True
                    max_pinnable_bytes = total_pinned_bytes + total
                    break

                total += size

            current_big_tensor = big_tensors[big_tensor_no]

            if is_buffer :
                _force_load_buffer(p) # otherwise potential memory leak
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                sub_offset = offset
                new_subs = {}
                for name, tensor in sub_tensors:
                    length = torch.numel(tensor) * tensor.element_size()
                    new_subs[name] = _move_to_pinned_tensor(tensor, current_big_tensor, sub_offset, length)
                    sub_offset += length
                _set_quantized_subtensors(p, new_subs)
                del new_subs, sub_tensors
            else:
                length = torch.numel(p.data) * p.data.element_size()
                p.data = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length)

            tensor_no += 1
        del p
    del dummy_pinned_tensor
    model._pinned_bytes = total
    total_pinned_bytes += total
    del params_dict
    gc.collect()

    if verboseLevel >=1:
        if partialPinning or failed_planned_allocation:        
            print(f"The model was partially pinned to reserved RAM: {last_allocated_big_tensor + 1} large blocks spread across {total/ONE_MB:.2f} MB")
        else:
            print(f"The whole model was pinned to reserved RAM: {last_allocated_big_tensor + 1} large blocks spread across {total/ONE_MB:.2f} MB")

    model._already_pinned = True


    return 
welcome_displayed = False

def _welcome():
    global welcome_displayed
    if welcome_displayed:
         return 
    welcome_displayed = True
    print(f"{BOLD}{HEADER}************ Memory Management for the GPU Poor (mmgp 3.6.10) by DeepBeepMeep ************{ENDC}{UNBOLD}")

def change_dtype(model, new_dtype, exclude_buffers = False):
    for submodule_name, submodule in model.named_modules():  
        if hasattr(submodule, "_lock_dtype"):
            continue
        for n, p in submodule.named_parameters(recurse = False):
            if p.data.dtype != new_dtype:
                p.data = p.data.to(new_dtype)

        if not exclude_buffers:
            for p in submodule.buffers(recurse=False):
                if p.data.dtype != new_dtype:
                    p.data = p.data.to(new_dtype)

    return model
            
def _extract_num_from_str(num_in_str):
    size = len(num_in_str)
    for i in range(size):
        if not num_in_str[-i-1:].isnumeric():
            if i == 0:
                return num_in_str, -1
            else:             
                return num_in_str[: -i],  int(num_in_str[-i:])                    
    return  "", -1 if size == 0 else int(num_in_str)

def  _quantize_dirty_hack(model):
    # dirty hack: add a hook on state_dict() to return a fake non quantized state_dict if called by Lora Diffusers initialization functions
    setattr( model, "_real_state_dict", model.state_dict)
    from collections import OrderedDict
    import traceback

    def state_dict_for_lora(self):
        real_sd = self._real_state_dict()
        fakeit = False
        stack = traceback.extract_stack(f=None, limit=5)
        for frame in stack:
            if "_lora_" in frame.name:
                fakeit = True
                break

        if not fakeit:
            return real_sd
        sd = OrderedDict()
        for k in real_sd:
            v = real_sd[k]
            if k.endswith("._data"):
                k = k[:len(k)-6]
            sd[k] = v
        return sd

    setattr(model, "state_dict", functools.update_wrapper(functools.partial(state_dict_for_lora, model), model.state_dict) )

def _quantization_map(model):
    from optimum.quanto import quantization_map
    return quantization_map(model)

def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)

def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights = None,
    activations = None,
    optimizer = None,
):
    
    qmodule = quantize_module(module, weights=weights, activations=activations, optimizer=optimizer)
    if qmodule is not None:
        _set_module_by_name(model, name, qmodule)
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param

def _requantize(model: torch.nn.Module, state_dict: dict, quantization_map: dict, default_dtype=None):
    # change dtype of current meta model parameters because 'requantize' won't update the dtype on non quantized parameters
    for k, p in model.named_parameters():
        if not k in quantization_map and k in state_dict:
            p_in_file = state_dict[k] 
            if p.data.dtype != p_in_file.data.dtype:
                p.data = p.data.to(p_in_file.data.dtype)

    # rebuild quanto objects
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)
            if default_dtype is not None:
                new_module = model.get_submodule(name)
                setter = getattr(new_module, "set_default_dtype", None)
                if callable(setter):
                    setter(default_dtype)

    model._quanto_map = quantization_map

    _quantize_dirty_hack(model)



def _quantize(model_to_quantize, weights=qint8, verboseLevel = 1, threshold = 2**31, model_id = 'Unknown'):
    
    total_size =0
    total_excluded = 0
    exclude_list = []
    submodule_size = 0
    submodule_names = []
    cur_blocks_prefix = None
    prev_blocks_prefix = None

    if hasattr(model_to_quantize, "_quanto_map"):
        for k, entry in model_to_quantize._quanto_map.items():
            weights  =  entry["weights"]
            print(f"Model '{model_id}' is already quantized to format '{weights}'")
            return False
        print(f"Model '{model_id}' is already quantized")
        return False

    print(f"Quantization of model '{model_id}' started to format '{weights}'")

    tower_names ,_  = _detect_main_towers(model_to_quantize)
    tower_names = [ n[:-1] for n in tower_names]


    cache_ref = {}
    tied_weights= {}
    reversed_tied_weights= {}

    for submodule_name, submodule in model_to_quantize.named_modules():  
        if isinstance(submodule, QModuleMixin):
            if verboseLevel>=1:
                print("No quantization to do as model is already quantized")
            return False

        size = 0
        for n, p in submodule.named_parameters(recurse = False):
            ref = _get_tensor_ref(p)
            match = cache_ref.get(ref, None)
            if match != None:
                tied_weights[submodule_name]=  (n, ) + match
                entries = reversed_tied_weights.get( match, [])
                reversed_tied_weights[match] = entries + [ (p, submodule_name,n)]
            else:
                cache_ref[ref] = (submodule_name, n)
                size  += torch.numel(p.data) * sizeofhalffloat

        for p in submodule.buffers(recurse=False):
            size  += torch.numel(p.data) * sizeofhalffloat

        already_added = False
        if hasattr(submodule, "_lock_dtype"):
            submodule_size += size
            submodule_names.append(submodule_name)
            already_added = True

        if not any(submodule_name.startswith(pre) for pre in tower_names):
            flush = False
            if cur_blocks_prefix == None or not submodule_name.startswith(cur_blocks_prefix):
                cur_blocks_prefix = submodule_name + "."
                flush = True                    

            if flush :
                if submodule_size <= threshold :
                    exclude_list += submodule_names
                    if verboseLevel >=2 and submodule_size >0:
                        print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
                    total_excluded += submodule_size

                submodule_size = 0
                submodule_names = []
            prev_blocks_prefix = cur_blocks_prefix
            if not already_added:
                submodule_size += size
                submodule_names.append(submodule_name)
        total_size += size

    if submodule_size >0  : 
        exclude_list += submodule_names
        if verboseLevel >=2:
            print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
        total_excluded += submodule_size


    perc_excluded =total_excluded/ total_size if total_size >0 else 1
    if verboseLevel >=2:
        if total_excluded == 0:
            print(f"Can't find any module to exclude from quantization, full model ({total_size/ONE_MB:.1f} MB) will be quantized")
        else:
            print(f"Total Excluded {total_excluded/ONE_MB:.1f} MB of {total_size/ONE_MB:.1f} that is {perc_excluded*100:.2f}%")
    if perc_excluded >= 0.20:
        if verboseLevel >=2:
            print(f"Too many modules are excluded, there is something wrong with the selection, switch back to full quantization.")
        exclude_list = None


    exclude_list += list(tied_weights) 
    quantize(model_to_quantize, weights= weights, exclude= exclude_list)


    # quantize(model_to_quantize,weights, include= [ "*1.block.attn.to_out*"]) #" 

    # for name, m in model_to_quantize.named_modules():
    #     if exclude_list is None or not any( name == module_name for module_name in exclude_list):
    #         _quantize_submodule(model_to_quantize, name, m, weights=weights, activations=None, optimizer=None)


    # force to read non quantized parameters so that their lazy tensors and corresponding mmap are released
    # otherwise we may end up keeping in memory both the quantized and the non quantize model
    named_modules = {n:m for n,m in model_to_quantize.named_modules()}

    for module_name, module in named_modules.items():
        # do not read quantized weights (detected them directly or behind an adapter)
        if isinstance(module, QModuleMixin) or hasattr(module, "base_layer") and  isinstance(module.base_layer, QModuleMixin): 
            if hasattr(module, "bias") and module.bias is not None:
                _force_load_parameter(module.bias)
        else:
            tied_w = tied_weights.get(module_name, None)
            for n, p in module.named_parameters(recurse = False):

                if tied_w != None and n == tied_w[0]:
                    if isinstance( named_modules[tied_w[1]], QModuleMixin) :
                        setattr(module, n, None) # release refs of tied weights if source is going to be quantized
                    # otherwise don't force load as it will be loaded in the source anyway
                else:
                    _force_load_parameter(p)
                    entries =  reversed_tied_weights.get( (module_name, n), [])
                    for tied_weight, tied_module_name, tied_weight_name in entries:
                        if n == tied_weight_name:
                             tied_weight.data = p.data

                del p #  del p if not it will still contain a ref to a tensor when leaving the loop
        for b in module.buffers(recurse = False):
            _force_load_buffer(b) 
            del b


    freeze(model_to_quantize)
    torch.cuda.empty_cache()
    gc.collect()       

    for tied_module, (tied_weight, src_module, src_weight) in tied_weights.items():  
        p = getattr(named_modules[src_module], src_weight)
        if isinstance(p, QTensor):
            setattr(named_modules[tied_module], tied_weight, p ) # copy refs to quantized sources

    del named_modules

    quantization_map = _quantization_map(model_to_quantize)

    model_to_quantize._quanto_map = quantization_map

    if hasattr(model_to_quantize, "_already_pinned"):
        delattr(model_to_quantize, "_already_pinned")

    _quantize_dirty_hack(model_to_quantize)

    print(f"Quantization of model '{model_id}' done")

    return True

def _as_field_tuple(value):
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _get_split_handler(info, field, default_handlers):
    handlers = info.get("split_handlers") or info.get("field_handlers") or {}
    if handlers:
        handler = handlers.get(field)
        if handler is not None:
            return handler
    if default_handlers:
        return default_handlers.get(field)
    return None


def _get_split_base_fields(info, split_fields):
    base_fields = _as_field_tuple(info.get("base_fields") or info.get("base_field"))
    if base_fields:
        return base_fields
    if split_fields:
        return (next(iter(split_fields.keys())),)
    return ()


def _merge_share_fields(info, share_fields):
    info_fields = _as_field_tuple(info.get("share_fields") or info.get("shared_fields"))
    return tuple(sorted(set(info_fields).union(_as_field_tuple(share_fields))))


def _call_split_handler(handler, *, src, dim, split_sizes, context):
    if handler is None:
        return None
    try:
        chunks = handler(src=src, dim=dim, split_sizes=split_sizes, context=context)
    except Exception:
        return None
    if not isinstance(chunks, (list, tuple)) or len(chunks) != len(split_sizes):
        return None
    return chunks


def _fill_sub_maps(sub_maps, name, value):
    for sub_map in sub_maps:
        sub_map[name] = value


def sd_split_linear(
    state_dict,
    split_map,
    split_fields=None,
    share_fields=None,
    verboseLevel=1,
    split_handlers=None,
):
    if not split_map:
        return state_dict
    split_fields = split_fields or {}
    share_fields = share_fields or ()
    split_handlers = split_handlers or {}
    base_fields_by_suffix = {
        suffix: _get_split_base_fields(info or {}, split_fields)
        for suffix, info in split_map.items()
    }
    def _skip(msg):
        if verboseLevel >= 2:
            print(f"[sd_split_linear] Skip {msg}")

    bases = {}
    for key in state_dict.keys():
        for suffix, base_fields in base_fields_by_suffix.items():
            for base_field in base_fields:
                suffix_token = f"{suffix}.{base_field}"
                if not key.endswith(suffix_token):
                    continue
                base = key[: -len("." + base_field)]
                if base.endswith(suffix):
                    bases[base] = suffix
                break

    if not bases:
        return state_dict

    for base, suffix in bases.items():
        info = split_map.get(suffix) or {}
        mapped = info.get("mapped_modules") or info.get("mapped_suffixes") or info.get("mapped") or []
        if not mapped:
            continue

        base_fields = base_fields_by_suffix.get(suffix) or _get_split_base_fields(info, split_fields)
        size_field = info.get("size_field") or (base_fields[0] if base_fields else None)
        size_tensor = state_dict.get(base + "." + size_field) if size_field else None
        split_dim = info.get("split_dim", 0)
        split_sizes = list(info.get("split_sizes") or [])
        if not split_sizes:
            if size_tensor is None:
                continue
            if size_tensor.dim() <= split_dim:
                _skip(f"{base}: dim={size_tensor.dim()} split_dim={split_dim}")
                continue
            out_dim = size_tensor.size(split_dim)
            if out_dim % len(mapped) != 0:
                _skip(f"{base}: out_dim={out_dim} not divisible by {len(mapped)}")
                continue
            split_sizes = [out_dim // len(mapped)] * len(mapped)
        elif None in split_sizes:
            if size_tensor is None:
                continue
            if size_tensor.dim() <= split_dim:
                _skip(f"{base}: dim={size_tensor.dim()} split_dim={split_dim}")
                continue
            known = sum(size for size in split_sizes if size is not None)
            none_count = split_sizes.count(None)
            remaining = size_tensor.size(split_dim) - known
            if remaining < 0 or remaining % none_count != 0:
                _skip(f"{base}: cannot resolve split sizes")
                continue
            fill = remaining // none_count
            split_sizes = [fill if size is None else size for size in split_sizes]

        total = sum(split_sizes)
        prefix = base[: -len(suffix)]
        target_bases = [prefix + name for name in mapped]
        added = 0

        field_tensors = {
            field: state_dict.get(base + "." + field)
            for field in set(split_fields.keys()).union(share_fields)
        }
        base_ctx = {
            "state_dict": state_dict,
            "base": base,
            "suffix": suffix,
            "split_sizes": split_sizes,
            "total": total,
            "mapped": mapped,
            "target_bases": target_bases,
            "verboseLevel": verboseLevel,
            "split_fields": split_fields,
            "share_fields": share_fields,
            "field_tensors": field_tensors,
            "size_field": size_field,
            "size_tensor": size_tensor,
            "split_dim": split_dim,
            "info": info,
        }
        fields_iter = list(split_fields.items()) + [(field, None) for field in share_fields]
        for field, dim in fields_iter:
            src = field_tensors.get(field)
            if src is None:
                continue
            if dim is None:
                for target_base in target_bases:
                    dest_key = target_base + "." + field
                    if dest_key not in state_dict:
                        state_dict[dest_key] = src
                        added += 1
                continue
            if src.dim() <= dim:
                _skip(f"{base}.{field}: dim={src.dim()} split_dim={dim}")
                continue
            if src.size(dim) != total:
                _skip(f"{base}.{field}: size({dim})={src.size(dim)} expected={total}")
                continue
            handler = _get_split_handler(info, field, split_handlers)
            chunks = _call_split_handler(
                handler,
                src=src,
                dim=dim,
                split_sizes=split_sizes,
                context=dict(base_ctx, field=field),
            )
            if chunks is None:
                chunks = torch.split(src, split_sizes, dim=dim)
            for target_base, chunk in zip(target_bases, chunks):
                if torch.is_tensor(chunk) and not chunk.is_contiguous():
                    chunk = chunk.contiguous()
                dest_key = target_base + "." + field
                if dest_key not in state_dict:
                    state_dict[dest_key] = chunk
                    added += 1

        if added:
            for field in list(split_fields.keys()) + list(share_fields):
                state_dict.pop(base + "." + field, None)
            if verboseLevel >= 2:
                print(f"[sd_split_linear] Split {base} -> {', '.join(mapped)}")

    return state_dict


def split_linear_modules(model, map, split_handlers=None, share_fields=None):
    from optimum.quanto import QModuleMixin
    from accelerate import init_empty_weights

    split_handlers = split_handlers or {}
    share_fields = share_fields or ()

    modules_dict = { k: m for k, m in model.named_modules()}
    for module_suffix, split_info in map.items():
        mapped_modules = split_info["mapped_modules"]
        split_sizes = split_info["split_sizes"]
        split_share_fields = _merge_share_fields(split_info, share_fields)
        split_dims = split_info.get("split_dims") or {}
        for k, module in modules_dict.items():
            if k.endswith("." + module_suffix):
                parent_module = modules_dict[k[:len(k)-len(module_suffix)-1]]
                weight = module.weight
                bias = getattr(module, "bias", None) 
                if isinstance(module, QModuleMixin):
                    out_features_total = weight.size(0)
                    if sum(split_sizes) != out_features_total:
                        raise ValueError(
                            f"Split sizes {split_sizes} do not match out_features {out_features_total} for '{k}'."
                        )
                    in_features = weight.size(1)
                    sub_biases = None
                    if bias is not None and bias.dim() > 0 and bias.size(0) == out_features_total:
                        sub_biases = torch.split(bias, split_sizes, dim=0)
                    else:
                        sub_biases = [bias] * len(split_sizes)

                    sub_tensors = _get_quantized_subtensors(weight)
                    if not sub_tensors:
                        raise ValueError(f"Unable to split quantized weight for '{k}'.")
                    sub_maps = [dict() for _ in split_sizes]
                    field_tensors = {name: tensor for name, tensor in sub_tensors}
                    base_ctx = {
                        "module": module,
                        "module_name": k,
                        "module_suffix": module_suffix,
                        "mapped_modules": mapped_modules,
                        "split_sizes": split_sizes,
                        "out_features": out_features_total,
                        "in_features": in_features,
                        "field_tensors": field_tensors,
                        "info": split_info,
                    }
                    for name, tensor in sub_tensors:
                        if tensor is None or name in split_share_fields or tensor.dim() <= 1:
                            _fill_sub_maps(sub_maps, name, tensor)
                            continue
                        split_dim = split_dims.get(name)
                        if split_dim is None:
                            if tensor.size(0) == out_features_total:
                                split_dim = 0
                            elif tensor.dim() > 1 and tensor.size(1) == out_features_total:
                                split_dim = 1
                            else:
                                split_dim = 0
                        handler = _get_split_handler(split_info, name, split_handlers)
                        chunks = _call_split_handler(
                            handler,
                            src=tensor,
                            dim=split_dim,
                            split_sizes=split_sizes,
                            context=dict(base_ctx, split_dim=split_dim),
                        )
                        if chunks is None:
                            if tensor.dim() <= split_dim or tensor.size(split_dim) != out_features_total:
                                got_size = "n/a" if tensor.dim() <= split_dim else tensor.size(split_dim)
                                raise ValueError(
                                    f"Cannot split '{k}' quantized tensor '{name}': "
                                    f"expected size({split_dim})={out_features_total}, got {got_size}."
                                )
                            chunks = torch.split(tensor, split_sizes, dim=split_dim)
                        for sub_map, chunk in zip(sub_maps, chunks):
                            sub_map[name] = chunk

                    create_fn = getattr(weight.__class__, "create", None)
                    if not callable(create_fn):
                        raise ValueError(f"Quantized weight class '{weight.__class__.__name__}' has no create()")
                    create_sig = inspect.signature(create_fn)
                    base_kwargs = {
                        "qtype": getattr(weight, "qtype", None),
                        "axis": getattr(weight, "axis", None),
                        "stride": weight.stride(),
                        "dtype": weight.dtype,
                        "activation_qtype": getattr(weight, "activation_qtype", None),
                        "requires_grad": weight.requires_grad,
                        "group_size": getattr(weight, "_group_size", None),
                        "device": weight.device,
                    }

                    qmodule_cls = module.__class__
                    for sub_name, sub_size, sub_map, sub_bias in zip(
                        mapped_modules, split_sizes, sub_maps, sub_biases
                    ):
                        with init_empty_weights():
                            sub_module = qmodule_cls(
                                in_features,
                                sub_size,
                                bias=bias is not None,
                                device="cpu",
                                dtype=weight.dtype,
                                weights=module.weight_qtype,
                                activations=module.activation_qtype,
                                optimizer=module.optimizer,
                                quantize_input=True,
                            )
                        size = list(weight.size())
                        if size:
                            size[0] = sub_size
                        base_kwargs["size"] = tuple(size)
                        create_kwargs = {}
                        missing = []
                        for name, param in create_sig.parameters.items():
                            if name == "self":
                                continue
                            if name in sub_map:
                                create_kwargs[name] = sub_map[name]
                            elif name in base_kwargs and base_kwargs[name] is not None:
                                create_kwargs[name] = base_kwargs[name]
                            elif param.default is param.empty:
                                missing.append(name)
                        if missing:
                            raise ValueError(
                                f"Unable to rebuild quantized weight for '{k}.{sub_name}': "
                                f"missing {missing}."
                            )
                        sub_weight = create_fn(**create_kwargs)
                        sub_module.weight = torch.nn.Parameter(sub_weight, requires_grad=weight.requires_grad)
                        if sub_bias is not None:
                            sub_module.bias = torch.nn.Parameter(sub_bias)
                        sub_module.optimizer = module.optimizer
                        sub_module.weight_qtype = module.weight_qtype
                        sub_module.activation_qtype = module.activation_qtype
                        setattr(parent_module, sub_name, sub_module)
                else:
                    sub_data = torch.split(weight, split_sizes, dim=0)
                    sub_bias = torch.split(bias, split_sizes, dim=0) if bias is not None else [None] * len(split_sizes)
                    for sub_name, subdata, subbias in zip(mapped_modules, sub_data, sub_bias):
                        with init_empty_weights():
                            sub_module = torch.nn.Linear( subdata.shape[1], subdata.shape[0], bias=bias != None, device ="cpu", dtype=weight.dtype)
                        sub_module.weight = torch.nn.Parameter(subdata , requires_grad=False)
                        if bias != None:
                            sub_module.bias = torch.nn.Parameter(subbias)
                        setattr(parent_module, sub_name, sub_module)

                delattr(parent_module, module_suffix)


def load_loras_into_model(model, lora_path, lora_multi = None, activate_all_loras = True, check_only = False, ignore_model_variations = False, pinnedLora = False, split_linear_modules_map = None, preprocess_sd = None, verboseLevel = -1,):
    verboseLevel = _compute_verbose_level(verboseLevel)

    loras_model_data = getattr(model, "_loras_model_data", None)
    if loras_model_data == None:
        raise Exception(f"No Loras has been declared for this model while creating the corresponding offload object")
    
    if not check_only:
        unload_loras_from_model(model)

    modules_dict = {k: v for k,v in model.named_modules()}

    CrLf = '\r\n'
    error_msg = ""
    def append(source, text ):
        if len(source) == 0:
            return text
        else:
            return source + CrLf + text
    
    def trunc(text, sz):
        text = str(text)
        if len(text) < sz:
            return text
        else:
            return text[0:sz] + '...'

    if not isinstance(lora_path, list):
        lora_path = [lora_path]
    
    if lora_multi is None:
        lora_multi = [1. for _ in lora_path]
    loras_nos = []
    loras_multi = []
    new_lora_path = []
    errors  = []
    adapters = {}
    adapter_no = 0
    pinned_sd_list = []
    pinned_names_list = []
    for i, path in enumerate(lora_path):
        adapter_name = str(adapter_no)
        error_msg = ""
        if not os.path.isfile(path):
            error_msg = f"Lora '{path}' was not found"
            errors.append((path, error_msg))
            print(error_msg)
            continue
        fail = False
        skip = False
        state_dict = safetensors2.torch_load_file(path, writable_tensors= False)

        if preprocess_sd != None:
            state_dict = preprocess_sd(state_dict)

        if split_linear_modules_map != None:
            new_state_dict = dict()
            suffixes = [(".alpha", -2, False), (".lora_B.weight", -3, True), (".lora_A.weight", -3, False), (".lora_up.weight", -3, True), (".lora_down.weight", -3, False),(".dora_scale", -2, False),]
            for module_name, module_data in state_dict.items():
                name_parts = module_name.split(".")
                for suffix, pos, any_split in suffixes: 
                    if module_name.endswith(suffix) and (map := split_linear_modules_map.get(name_parts[pos], None )) != None:
                        parent_module_name, module_name = ".".join(name_parts[:pos]), None
                        sub_data = torch.split(module_data, map["split_sizes"], dim=0) if any_split else [None] * len(map["mapped_modules"])  
                        for sub_name, subdata in zip(map["mapped_modules"], sub_data):
                            new_module_name = parent_module_name + "." + sub_name + suffix
                            new_state_dict[new_module_name] = subdata if any_split else module_data
                        break
                if module_name != None: new_state_dict[module_name] = module_data            
            state_dict = new_state_dict
            del new_state_dict
            # tied_weights = _extract_tie_weights_from_sd(state_dict, path) # to do

        clean_up = False
        first_key = next(iter(state_dict), None)
        if first_key == None:
            msg = f"Empty Lora '{path}'"
            error_msg = append(error_msg, msg) 
            fail = True

        if not fail:
            pos = first_key.find(".")
            prefix = first_key[0:pos+1]
            if prefix in ["diffusion_model.", "transformer."]:
                state_dict = { k[ len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix) }

            clean_up = True

            keys = list(state_dict.keys())

            lora_alphas = {}
            for k in keys:
                if k.endswith(".alpha"):
                    alpha_value = state_dict.pop(k)
                    if torch.is_tensor(alpha_value):
                        alpha_value = float(alpha_value.item())
                    lora_alphas[k] = alpha_value

            invalid_keys = []
            unexpected_keys = []
            new_state_dict = {}
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                lora_A = lora_B = diff_b = diff = lora_key = dora_scale = None
                if k.endswith(".diff"):
                    diff = v
                    module_name = k[ : -5]
                elif k.endswith(".diff_b"):
                    diff_b = v
                    module_name = k[ : -7]
                elif k.endswith(".dora_scale"):
                    dora_scale = v
                    module_name = k[ : -11]
                else:
                    pos = k.rfind(".lora_")
                    if pos <=0:
                        invalid_keys.append(k)
                        continue
                    module_name = k[ : pos]
                    lora_key = k[ pos+1:]
                    if lora_key in ("lora_A.weight", "lora_down.weight"):
                        lora_A = v
                    elif lora_key in ("lora_B.weight", "lora_up.weight"):
                        lora_B = v
                    else:
                        invalid_keys.append(k)
                        continue

                module =  modules_dict.get(module_name, None)
                if module == None:
                    unexpected_keys.append(k)
                    continue
                if False: #not isinstance(module, (QLinear, torch.nn.Linear, torch.nn.Conv3d, torch.nn.LayerNorm)):
                    msg = f"Lora '{path}' contains a non supported type of layer '{k}'"
                    error_msg = append(error_msg, msg) 
                    fail = True
                    break
                module_shape = module.weight.shape
                rank = None
                if lora_A != None:
                    rank = lora_A.shape[0] 
                    if module_shape[1] != v.shape[1]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora A dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[1]}, lora A = {v.shape[1]}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_A = lora_A.to(module.weight.dtype)                     
                elif lora_B != None:
                    rank = lora_B.shape[1] 
                    if module_shape[0] != v.shape[0]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora B dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, lora B = {v.shape[0]}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_B = lora_B.to(module.weight.dtype)                     
                elif diff != None:
                    lora_B = diff
                    if module_shape != v.shape:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora shape is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lora = {v.shape}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_B = lora_B.to(module.weight.dtype)                     
                elif diff_b != None:
                    rank = diff_b.shape[0] 
                    if not hasattr(module, "bias"):
                        pass
                    if module.bias == None:
                        msg = f"Lora '{path}': Lora Basis is defined while it doesnt exist in model '{_get_module_name(model)}'. It is likely this Lora has been made for another version of this model."
                        fail = True
                        break
                    else:
                        module_shape = module.bias.shape
                        if module_shape != v.shape:
                            if ignore_model_variations:
                                skip = True
                            else:
                                msg = f"Lora '{path}': Lora Basis dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, lora Basis = {v.shape[0]}). It is likely this Lora has been made for another version of this model."
                                error_msg = append(error_msg, msg) 
                                fail = True
                            break
                    v = diff_b = diff_b.to(module.weight.dtype)                     
                elif dora_scale != None:
                    rank = dora_scale.shape[1] 
                    if module_shape[0] != v.shape[0]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}': Dora Scale dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, dora scale = {v.shape[0]}). It is likely this Dora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = dora_scale = dora_scale.to(module.weight.dtype)                     
                if not check_only:
                    new_state_dict[k] = v
                    v = None
                    loras_module_data = loras_model_data.get(module, None)
                    assert loras_module_data != None
                    loras_adapter_data =  loras_module_data.get(adapter_name, None)
                    if loras_adapter_data == None:
                        loras_adapter_data = [None, None, None, None, 1.]
                        module.any_dora = False
                        loras_module_data[adapter_name] = loras_adapter_data
                    if lora_A != None:
                        loras_adapter_data[0] = lora_A
                    elif lora_B != None:
                        loras_adapter_data[1] = lora_B 
                    elif dora_scale != None:
                        loras_adapter_data[3] = dora_scale 
                        loras_module_data["any_dora"] = True
                    else:
                        loras_adapter_data[2] = diff_b 
                    if rank != None and lora_key is not None and "lora" in lora_key:
                        alpha_key = k[:-len(lora_key)] + "alpha"
                        alpha = lora_alphas.get(alpha_key, None)
                        if alpha is not None: loras_adapter_data[4] = alpha / rank 
            lora_A = lora_B = diff = diff_b = v = loras_module_data = loras_adapter_data = lora_alphas = dora_scale = None

            if len(invalid_keys)  > 0:
                msg = f"Lora '{path}' contains non Lora keys '{trunc(invalid_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
            if len(unexpected_keys)  > 0:
                msg = f"Lora '{path}' contains unexpected module keys, it is likely that this Lora is for a different model : '{trunc(unexpected_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
        if fail or skip:
            if fail:
                errors.append((path, error_msg))
                print(error_msg)
            if clean_up and not check_only:
                for m,loras_module_data in loras_model_data.items():
                    if adapter_name in loras_module_data:
                        del loras_module_data[adapter_name]

        else:
            if not check_only:
                # model._loras_tied_weights[adapter_name] = tied_weights
                if pinnedLora:
                    pinned_sd_list.append(new_state_dict)
                    pinned_names_list.append(path)
                    # _pin_sd_to_memory(state_dict, path)

            del state_dict 


            adapters[adapter_name] = path
            loras_nos.append(adapter_name)
            new_lora_path.append(path)        
            loras_multi.append(1.0 if i > (len(lora_multi) -1) else lora_multi[i])
            pass
            adapter_no += 1
            if verboseLevel >=1:
                if check_only:
                    print(f"Lora '{path}' was found for model '{_get_module_name(model)}'")
                else:
                    print(f"Lora '{path}' was loaded in model '{_get_module_name(model)}'")
    
    model._loras_errors = errors
    if not check_only:
        if pinnedLora and len(pinned_sd_list) > 0:
            _pin_sd_to_memory(pinned_sd_list, pinned_names_list)
        model._loras_adapters = adapters
    if activate_all_loras:
        activate_loras(model, loras_nos, loras_multi)
    return new_lora_path


def merge_dicts(A, B):
    for key, value in A.items():
        if isinstance(value, dict):
            if key not in B or not isinstance(B[key], dict):
                B[key] = value  # Copy entire dict reference from A
            else:
                merge_dicts(value, B[key])  # Recurse into both dicts
        else:
            B[key] = value  # Copy non-dict value from A to B


def sync_models_loras(model, model2):
    merge_dicts(model._loras_model_shortcuts , model2._loras_model_shortcuts)
    model2._loras_active_adapters = model._loras_active_adapters 
    model2._loras_adapters = model._loras_adapters
    model2._loras_scaling = model._loras_scaling 

def unload_loras_from_model(model):
    if model is None: return
    if not hasattr(model, "_loras_model_data"): return
    for _, v in model._loras_model_data.items():
        v.clear()
    for _, v in model._loras_model_shortcuts.items():
        v.clear()

    model._loras_active_adapters = []
    model._loras_scaling = dict()
    model._loras_tied_weights = dict()
    model._loras_errors = None
    model._loras_adapters = None
    model._loras_scaling = None


def set_step_no_for_lora(model, step_no):
    model._lora_step_no = step_no

def activate_loras(model, lora_nos, lora_multi = None):
    if not isinstance(lora_nos, list):
        lora_nos = [lora_nos]
    lora_nos = [str(l) for l in lora_nos]

    if lora_multi is None:
        lora_multi = [1. for _ in lora_nos]

    lora_scaling_dict = {}
    for no, multi in zip(lora_nos, lora_multi):
        lora_scaling_dict[no] = multi

    model._lora_step_no = 0    
    model._loras_active_adapters = lora_nos
    model._loras_scaling = lora_scaling_dict 


def move_loras_to_device(model, device="cpu" ):
    if hasattr( model, "_lora_loadable_modules"):
        for k in model._lora_loadable_modules:
            move_loras_to_device(getattr(model,k), device)
        return
    
    for k, m in model.named_modules():
        if ".lora_" in k:
            m.to(device)

def fast_load_transformers_model(model_path: str,  do_quantize = False, quantizationType =  qint8, pinToMemory = False, partialPinning = False, forcedConfigPath = None, defaultConfigPath = None, modelClass=None, modelPrefix = None, writable_tensors = True, verboseLevel = -1, preprocess_sd  = None, modules = None,  return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, configKwargs ={}):
    """
    quick version of .LoadfromPretrained of  the transformers library
    used to build a model and load the corresponding weights (quantized or not)
    """       

    
    import os.path
    if not isinstance(model_path, list):
        model_path = [model_path]


    if not builtins.all(file_name.endswith(".sft") or file_name.endswith(".safetensors") or file_name.endswith(".pt") or file_name.endswith(".ckpt") for file_name in model_path):
        raise Exception("full model path to file expected")

    model_path = [ _get_model(file) for file in model_path] 
    if any( file == None for file in model_path):
        raise Exception("Unable to find file")
    
    verboseLevel = _compute_verbose_level(verboseLevel)
    if model_path[-1].endswith(".pt") or model_path[-1].endswith(".ckpt"):
        metadata = None
    else:
        with safetensors2.safe_open(model_path[-1], writable_tensors =writable_tensors) as f:
            metadata = f.metadata() 

    if metadata is None:
        transformer_config = None
    else:
        transformer_config = metadata.get("config", None)

    if transformer_config == None or forcedConfigPath != None:
        if forcedConfigPath != None:
            config_fullpath = forcedConfigPath
        else:
            config_fullpath =  os.path.join(os.path.dirname(model_path[-1]), "config.json") if defaultConfigPath == None else defaultConfigPath

        if not os.path.isfile(config_fullpath):
            raise Exception("a 'config.json' that describes the model is required in the directory of the model or inside the safetensor file")

        with open(config_fullpath, "r", encoding="utf-8") as reader:
            text = reader.read()
        transformer_config= json.loads(text)

    transformer_config.update( configKwargs )

    if "architectures" in transformer_config: 
        architectures = transformer_config["architectures"]
        class_name = architectures[0] 
        if modelClass !=None:
            transfomer_class = modelClass
        else:
            module = __import__("transformers")
            map = {  "T5WithLMHeadModel" : "T5EncoderModel"}
            class_name = map.get(class_name, class_name)
            transfomer_class = getattr(module, class_name)
        from transformers import AutoConfig

        import tempfile
        with tempfile.NamedTemporaryFile("w", delete = False,  encoding ="utf-8") as fp: 
            fp.write(json.dumps(transformer_config))
            fp.close()
            config_obj = AutoConfig.from_pretrained(fp.name)     
        os.remove(fp.name)
        #needed to keep inits of non persistent buffers
        with init_empty_weights():
            model = transfomer_class(config_obj)
                

    else:
        if modelClass !=None:
            transfomer_class = modelClass
        elif "_class_name" in transformer_config:
            class_name  = 'Transformer3DModel'
            module = __import__("diffusers")
            transfomer_class = getattr(module, class_name)
        else:
            raise Exception("class not defined")                

        with init_empty_weights():
            model = transfomer_class.from_config(transformer_config )


    model.eval().requires_grad_(False)

    model._config = transformer_config

    load_model_data(model,model_path, do_quantize = do_quantize, quantizationType = quantizationType, pinToMemory= pinToMemory, partialPinning= partialPinning, modelPrefix = modelPrefix, writable_tensors =writable_tensors, preprocess_sd = preprocess_sd , modules = modules, return_shared_modules =  return_shared_modules, default_dtype = default_dtype, ignore_unused_weights = ignore_unused_weights, verboseLevel=verboseLevel )

    return model



@cudacontext("cpu")
def load_model_data(model, file_path, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, modelPrefix = None, writable_tensors = True,  preprocess_sd = None, postprocess_sd = None, modules = None, return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, verboseLevel = -1):
    """
    Load a model, detect if it has been previously quantized using quanto and do the extra setup if necessary
    """


    def filter_state_dict(state_dict, base_model_prefix):
        new_state_dict= {}
        start = -1
        for k,v in state_dict.items():
            if k.startswith(base_model_prefix):

                new_start = len(base_model_prefix)
            else:
                pos = k.find("." + base_model_prefix)
                if pos < 0:
                    continue
                new_start = pos + len(base_model_prefix)  +1
            if start != -1 and start != new_start:
                new_state_dict  = state_dict
                break
            start = new_start  
            new_state_dict[k[ start:]] = v
        return new_state_dict



    if not isinstance(file_path, list):
        file_path = [file_path]

    file_count =  len(file_path)
    if isinstance(modules, (list,str)):
        if isinstance(modules, str): modules = [modules]
        file_path += modules
        modules = None

    file_path = [ _get_model(file) for file in file_path] 
    if any( file == None for file in file_path):
        raise Exception("Unable to find file")
    verboseLevel = _compute_verbose_level(verboseLevel)

    model = _remove_model_wrapper(model)

    if return_shared_modules is not None:
        return_state_dict ={}
        return_quantization_map ={}
        return_shared_modules["state_dict"] = return_state_dict 
        return_shared_modules["quantization_map"] = return_quantization_map 

    full_quantization_map = {}
    full_tied_weights_map = {}
    full_state_dict = {}
    for no, file in enumerate(file_path):
        quantization_map = None
        tied_weights_map = None
        metadata = None
        detected_kind = None
        if not (".safetensors" in file or ".sft" in file): 
            if pinToMemory:
                raise Exception("Pinning to memory while loading only supported for safe tensors files")
            state_dict = torch.load(file, weights_only=False, map_location="cpu")
            if "module" in state_dict:
                state_dict = state_dict["module"]
            
        else:
            basename = os.path.basename(file)

            if "-of-" in basename:
                file_parts= basename.split("-")
                parts_max = int(file_parts[-1][:5])
                state_dict = {}
                for i in range(1, parts_max + 1):
                    file_parts[1] = ("0000" + str(i))[:5]
                    sd, _ = _safetensors_load_file( os.path.join( os.path.dirname(file), "-".join(file_parts) ) , writable_tensors =writable_tensors)
                    state_dict.update(sd)
            else:
                state_dict, metadata = _safetensors_load_file(file, writable_tensors =writable_tensors)

        if preprocess_sd != None:
            state_dict = preprocess_sd(state_dict)

        if metadata !=  None:
            quantization_map = metadata.get("quantization_map", None)
            config = metadata.get("config", None)
            if config is not None:
                model._config = config

            tied_weights_map = metadata.get("tied_weights_map", None)
            if tied_weights_map != None:
                for name, tied_weights_list in tied_weights_map.items():
                    mapped_weight = state_dict[name]
                    for tied_weights in tied_weights_list:
                        state_dict[tied_weights] = mapped_weight

        if quantization_map is None:
            pos = str.rfind(file, ".")
            if pos > 0:
                quantization_map_path = file[:pos]
            quantization_map_path += "_map.json"

            if os.path.isfile(quantization_map_path):
                with open(quantization_map_path, 'r') as f:
                    quantization_map = json.load(f)

        if quantization_map is None:
            conv_result = detect_and_convert(state_dict, default_dtype=default_dtype, verboseLevel=verboseLevel)
            detected_kind = conv_result.get("kind")
            if conv_result.get("kind") not in ("none", "quanto"):
                state_dict = conv_result["state_dict"]
                quantization_map = conv_result["quant_map"]
                conv_result = None
                # enable_fp8_fp32_scale_support()

            if detected_kind in (None, "none") and (".safetensors" in file or ".sft" in file):
                try:
                    info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel)
                    detected_kind = info.get("kind")
                except Exception:
                    detected_kind = detected_kind or None
            if detected_kind not in (None, "none"):
                cache_quantization_for_file(file, detected_kind or "none")
        
        full_state_dict.update(state_dict)
        if quantization_map != None:
            full_quantization_map.update(quantization_map)
        if tied_weights_map != None:
            full_tied_weights_map.update(tied_weights_map)
        if return_shared_modules is not None and no >= file_count:
            return_state_dict.update(state_dict)
            if quantization_map is not None: return_quantization_map.update(quantization_map)

    if isinstance(modules, dict) :
        full_state_dict.update(modules["state_dict"])
        full_quantization_map.update(modules["quantization_map"])

    state_dict, quantization_map, tied_weights_map  = full_state_dict, full_quantization_map, full_tied_weights_map
    full_state_dict, full_quantization_map, full_tied_weights_map = None, None, None

    # deal if we are trying to load just a sub part of a larger model
    if postprocess_sd != None:
        state_dict, quantization_map = postprocess_sd(state_dict, quantization_map)
        
    if modelPrefix != None:
        base_model_prefix = modelPrefix + "."
        state_dict = filter_state_dict(state_dict,base_model_prefix)
        if quantization_map != None:
            quantization_map = filter_state_dict(quantization_map,base_model_prefix)

    post_load_hooks = []
    if quantization_map:
        quantization_map, post_load_hooks = apply_pre_quantization(
            model,
            state_dict,
            quantization_map,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
        )

    if len(quantization_map) == 0:
        if any("quanto" in file for file in file_path) and not do_quantize:
            print("Model seems to be quantized by quanto but no quantization map was found whether inside the model or in a separate '{file_path[:json]}_map.json' file")
    else:
        _requantize(model, state_dict, quantization_map, default_dtype=default_dtype)    



    missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
    if len(missing_keys) > 0  :
        # if there is a key mismatch maybe we forgot to remove some prefix
        base_model_prefix = None
        for k,v in state_dict.items():
            if k.endswith(missing_keys[0]):
                base_model_prefix = k[:-len(missing_keys[0])]
                break
        if base_model_prefix == None:
            raise Exception(f"Missing keys: {missing_keys}")
        state_dict = filter_state_dict(state_dict, base_model_prefix)
        missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
        
    del state_dict

    if post_load_hooks:
        for hook in post_load_hooks:
            try:
                hook(model)
            except Exception as e:
                if verboseLevel >= 2:
                    print(f"Post-load hook skipped: {e}")

    if len(unexpected_keys) > 0 and verboseLevel >=2 and not ignore_unused_weights:
        print(f"Unexpected keys while loading '{file_path}': {unexpected_keys}")

    for k,p in model.named_parameters():
        if p.is_meta :
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since parameter '{k}' has no data"
            raise Exception(txt)
    for k,b in model.named_buffers():
        if b.is_meta :
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since buffer '{k}' has no data"
            raise Exception(txt)
        
    if return_shared_modules is not None:
        mods = { k : v for k,v in model.named_modules()}
        return_parameters = {}
        return_shared_modules["parameters"] = return_parameters
        for k in return_state_dict:
            if k.endswith("._data"):
                k = k[:-6]
            pos = k.rfind(".")
            mod_name = k[:pos]
            param_name =  k[pos +1:]
            mod = mods.get(mod_name, None)
            if mod is not None:
                p =  mod._parameters.get(param_name, None)
                if p is None: p =  mod._buffers.get(param_name, None)
                if p is not None:
                    return_parameters[k] = p
        del mods
        
    if isinstance(modules, dict) :
        mods = { k : v for k,v in model.named_modules()}
        # replace Parameter outer shell so that both models parameters are tied
        for k, rep_p in modules["parameters"].items():
            pos = k.rfind(".")
            mod_name = k[:pos]
            param_name =  k[pos +1:]
            mod = mods.get(mod_name, None)
            if mod is not None:
                setattr(mod, param_name, rep_p)
        del mods 
        modules["parameters"].clear()
        modules["state_dict"].clear()
        rep_p = p = None

    if do_quantize:
        if quantization_map != None and len(quantization_map) > 0 :
            if verboseLevel >=1:
                print("Model already quantized")
        else:
            if _quantize(model, quantizationType, verboseLevel=verboseLevel, model_id=file_path):
                quantization_map = model._quanto_map  

    if pinToMemory:
        _pin_to_memory(model, file_path, partialPinning = partialPinning, verboseLevel = verboseLevel)

    return

def save_model(model, file_path, do_quantize = False, quantizationType = qint8, verboseLevel = -1, config_file_path = None, filter_sd =None ):
    """save the weights of a model and quantize them if requested
    These weights can be loaded again using 'load_model_data'
    """       
    
    config = None
    extra_meta = None
    verboseLevel = _compute_verbose_level(verboseLevel)
    if config_file_path !=None:
        with open(config_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
            config= json.loads(text)
    elif hasattr(model, "_config"):
        config = model._config
    elif hasattr(model, "config"):
        config_fullpath = None
        config_obj = getattr(model,"config")
        config_path = getattr(config_obj,"_name_or_path", None)
        if config_path != None:
            config_fullpath = os.path.join(config_path, "config.json")      
            config_fullpath = _get_model(config_fullpath)

            # if not os.path.isfile(config_fullpath):
            #     config_fullpath = None
        if config_fullpath is None:                            
            config_fullpath =  os.path.join(os.path.dirname(file_path), "config.json")
        if os.path.isfile(config_fullpath):
            with open(config_fullpath, "r", encoding="utf-8") as reader:
                text = reader.read()
                config= json.loads(text)

    if do_quantize:
        _quantize(model, weights=quantizationType, model_id=file_path, verboseLevel=verboseLevel)
    
    quantization_map = getattr(model, "_quanto_map", None)

    from collections import OrderedDict

    cache_ref = {}
    tied_weights_map = {}
    sd = model.state_dict()
    if filter_sd  != None:
        new_sd = {}
        new_quantization_map = {}
        for k_k in filter_sd:
            for s in [".weight", ".bias", ".weight._data", ".weight._scale"]:                
                if k_k.endswith(s): 
                    k_k= k_k[:-len(s)]
                    break
            for k,v in sd.items():
                if k.startswith(k_k):
                    new_sd[k] = v
            if quantization_map != None:
                for k,v in quantization_map.items():
                    if k.startswith(k_k):
                        new_quantization_map[k] = v
        sd = new_sd
        if quantization_map != None: quantization_map = new_quantization_map

    out_sd = OrderedDict()


    for name, weight  in sd.items():
        ref = _get_tensor_ref(weight)
        match = cache_ref.get(ref, None)
        if match != None:
            tied_list = tied_weights_map.get(match, [])
            tied_list.append(name)
            tied_weights_map[match] = tied_list 
        else:
            out_sd[name] = weight 
            cache_ref[ref] = name

    if len(tied_weights_map) > 0:
        extra_meta = { "tied_weights_map" : tied_weights_map }

    if verboseLevel >=1:
        print(f"Saving file '{file_path}")

    safetensors2.torch_write_file(out_sd,  file_path , quantization_map = quantization_map, config = config, extra_meta= extra_meta)
    if verboseLevel >=1:
        print(f"File '{file_path}' saved")


def extract_models(obj = None, prefix = None):
    if isinstance(obj, str): # for compatibility as the two args were switched
        bkp = prefix
        prefix = obj
        obj = bkp

    pipe = {}
    if obj == None:
        raise Exception("an object to analyze must be provided")
    if prefix==None or len(prefix)==0:
        prefix = ""
    elif prefix[ -1:] != "/":
        prefix  + "/"        
    
    for name in dir(obj):
        if name in ["_execution_device"]:
            continue            
        element = getattr(obj,name)
        if name  in ("pipeline", "pipe"):
            pipeline = element
            if  hasattr(pipeline , "components") and isinstance(pipeline.components, dict):
                for k, model in pipeline.components.items():
                    if model != None:
                        pipe[prefix  + k ] = model
        elif isinstance(element, torch.nn.Module) and name!="base_model": 
            if prefix + name in pipe:
                pipe[prefix + "_" + name ] = element
            else:
                pipe[prefix + name ] = element
        elif isinstance(element, dict):
            for k, element in element.items():
                if  hasattr(element , "pipeline"):
                    pipe.update( extract_models(prefix + k,element ))


    return pipe

def get_model_name(model):
    return model.name

class HfHook:
    def __init__(self):
        self.execution_device = "cuda"

    def init_hook(self, module):
        return module

    def detach_hook(self, module):
        return module
    
last_offload_obj = None
class offload:
    def __init__(self):
        self.active_models = []
        self.active_models_ids = []
        self.models = {}
        self.cotenants_map = { 
                            "text_encoder": ["vae", "text_encoder_2"],
                            "text_encoder_2": ["vae", "text_encoder"],                             
                        }
        self.verboseLevel = 0
        self.blocks_of_modules = {}
        self.blocks_of_modules_sizes = {}
        self.anyCompiledModule = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check =0
        self.loaded_blocks = {}
        self.prev_blocks_names = {}
        self.next_blocks_names = {}
        self.preloaded_blocks_per_model = {}
        self.default_stream = torch.cuda.default_stream(torch.device("cuda")) # torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        self.async_transfers = False
        self.parameters_ref  = {} 
        self.max_reservable_memory = 0

        global last_offload_obj
        last_offload_obj = self

        self._type_wrappers = {}
        
    def add_module_to_blocks(self, model_id, blocks_name, submodule, prev_block_name, submodule_name):

        if blocks_name!=None and ".lora_" in blocks_name:
            blocks_name = None
        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if entry_name in self.blocks_of_modules:
            blocks_params = self.blocks_of_modules[entry_name]
            blocks_params_size = self.blocks_of_modules_sizes[entry_name]
        else:
            blocks_params = []
            self.blocks_of_modules[entry_name] = blocks_params
            blocks_params_size = 0
            if blocks_name !=None:
                prev_entry_name = None if prev_block_name == None else  model_id + "/" + prev_block_name
                self.prev_blocks_names[entry_name] =  prev_entry_name
                if not prev_block_name == None:
                    self.next_blocks_names[prev_entry_name] = entry_name        
        bef = blocks_params_size

        for k,p in submodule.named_parameters(recurse=False):
            param_size = 0
            ref = _get_tensor_ref(p)
            tied_param =  self.parameters_ref.get(ref, None)
            blocks_params.append((submodule, k, p, False, tied_param))
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                param_size += _subtensors_nbytes(sub_tensors)
                del sub_tensors
            else:
                param_size += torch.numel(p.data) * p.data.element_size()


            if tied_param == None:
                blocks_params_size +=  param_size
                self.parameters_ref[ref] = (submodule, k)

        for k, p in submodule.named_buffers(recurse=False):
            blocks_params.append( (submodule, k, p, True, None) )
            blocks_params_size += p.data.nbytes

        aft = blocks_params_size

        # if blocks_name is None:
        #     print(f"Default: {model_id}/{submodule_name} : {(aft-bef)/ONE_MB:0.2f} MB")
        #     pass


        self.blocks_of_modules_sizes[entry_name] = blocks_params_size


        return blocks_params_size


    def can_model_be_cotenant(self, model_id):
        potential_cotenants= self.cotenants_map.get(model_id, None)
        if potential_cotenants is None: 
            return False
        for existing_cotenant in self.active_models_ids:
            if existing_cotenant not in potential_cotenants: 
                return False    
        return True

    def _move_loras(self, loras_active_adapters, loras_modules,  to_GPU):
        for name, lora_module in loras_modules.items():
            for adapter in loras_active_adapters:
                lora_data = lora_module.get(adapter, None)
                if lora_data == None:
                    continue                     
                key = adapter + '_GPU'
                if to_GPU:
                    lora_module[key] = [None if item == None else item.cuda(non_blocking=True) for item in lora_data[ :-1] ] + lora_data[ -1:] 
                elif key in lora_module:
                    del lora_module[key]
            
    @torch.compiler.disable()
    def gpu_load_blocks(self, model_id, blocks_name, preload = False):
        # cl = clock.start()


        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        
        def cpu_to_gpu(stream_to_use, blocks_params): #, record_for_stream = None
            model = self.models[model_id]
            loras_modules = {}
            loras_active_adapters =  getattr(model ,"_loras_active_adapters", None)
            if loras_active_adapters == None or len(loras_active_adapters) == 0:
                loras_model_data = None
            else:
                loras_model_data =  getattr(model, "_loras_model_data", None)

            with torch.cuda.stream(stream_to_use):
                for param in blocks_params:
                    parent_module, n, p, is_buffer, tied_param = param

                    if tied_param != None:
                        tied_p = getattr( tied_param[0], tied_param[1]) 
                        if tied_p.is_cuda:
                            setattr(parent_module, n , tied_p)
                            continue
                    # if hasattr(p,'_data'):
                    #     if not p._data.is_pinned() or not p._scale.is_pinned():
                    #         pass
                    # else:
                    #     if  not p.data.is_pinned():
                    #         pass

                    q = p.to("cuda", non_blocking=True)
                    if is_buffer:
                        q = torch.nn.Buffer(q)
                    else:
                        q = torch.nn.Parameter(q , requires_grad=False)
                    setattr(parent_module, n , q)

                    if tied_param != None:
                        setattr( tied_param[0], tied_param[1], q) 
                    del p, q
                    if loras_model_data != None:
                        lora_data =  loras_model_data.get(parent_module, None)
                        if lora_data != None:
                            loras_modules[parent_module]= lora_data
                if len(loras_modules) > 0:
                    self._move_loras(loras_active_adapters, loras_modules, True)

        loaded_block = self.loaded_blocks[model_id]

        if not preload and loaded_block != None:
            self.gpu_unload_blocks(model_id, loaded_block)
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()


        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            # if not preload:
            #     print(f"Request to load model {entry_name} ({model_name}) in GPU")
                

        if self.async_transfers and blocks_name != None:
            prev = self.prev_blocks_names[entry_name]
            first = prev == None or prev != loaded_block
            next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
            if first:
                if self.verboseLevel >=2:
                    if preload:
                        print(f"Preloading model {entry_name} ({model_name}) in GPU")
                    else:
                        print(f"Loading model {entry_name} ({model_name}) in GPU")
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules[entry_name])

            torch.cuda.synchronize()

            if next_blocks_entry != None:
                if self.verboseLevel >=2:
                    print(f"Prefetching model {next_blocks_entry} ({model_name}) in GPU")
                cpu_to_gpu(self.transfer_stream, self.blocks_of_modules[next_blocks_entry]) #, self.default_stream

        else:
            if self.verboseLevel >=2:
                print(f"Loading model {entry_name} ({model_name}) in GPU")
            cpu_to_gpu(self.default_stream, self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()
        if not preload:
            self.loaded_blocks[model_id] = blocks_name           

        # cl.stop()
        # print(f"load time: {cl.format_time_gap()}")

    @torch.compiler.disable()
    def gpu_unload_blocks(self, model_id, blocks_name):
        # cl = clock.start()
        if blocks_name != None and blocks_name == self.loaded_blocks[model_id]:
            self.loaded_blocks[model_id] = None 


        blocks_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            print(f"Unloading model {blocks_name} ({model_name}) from GPU")
 
        blocks_params = self.blocks_of_modules[blocks_name]
        model = self.models[model_id]
        loras_modules = {}
        loras_active_adapters =  getattr(model ,"_loras_active_adapters", None)
        if loras_active_adapters == None or len(loras_active_adapters) == 0 :
            loras_model_data = None
        else:
            loras_model_data =  getattr(model, "_loras_model_data", None)

        for param in blocks_params:
            parent_module, n, p, is_buffer, _  = param
            if is_buffer:
                q = torch.nn.Buffer(p)
            else:
                q = torch.nn.Parameter(p , requires_grad=False)
            setattr(parent_module, n , q)
            del p, q 

            if loras_model_data != None:
                lora_data =  loras_model_data.get(parent_module, None)
                if lora_data != None:
                    loras_modules[parent_module]= lora_data

        if len(loras_modules) > 0:
            self._move_loras(loras_active_adapters, loras_modules, False)

        # cl.stop()
        # print(f"unload time: {cl.format_time_gap()}")

    # @torch.compiler.disable()
    def gpu_load(self, model_id):
        model = self.models[model_id]
        self.active_models.append(model)
        self.active_models_ids.append(model_id)
        self.gpu_load_blocks(model_id, None, True)
        for block_name in self.preloaded_blocks_per_model[model_id]:
            self.gpu_load_blocks(model_id, block_name, True)

    def unload_all(self):
        for model_id in self.active_models_ids:
            self.gpu_unload_blocks(model_id, None)      
            for block_name in self.preloaded_blocks_per_model[model_id]:
                self.gpu_unload_blocks(model_id, block_name)

            loaded_block = self.loaded_blocks[model_id]
            if loaded_block != None:
                self.gpu_unload_blocks(model_id, loaded_block)
                entry_name = model_id + "/" + loaded_block
                next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
                if next_blocks_entry != None:
                    pos = next_blocks_entry.rfind("/")
                    torch.cuda.synchronize()
                    self.gpu_unload_blocks(model_id, next_blocks_entry[pos+1:])      
                self.loaded_blocks[model_id] = None  
 
        self.active_models = []
        self.active_models_ids = []
        torch.cuda.empty_cache()
        gc.collect()
        self.last_reserved_mem_check = time.time()

    def move_args_to_gpu(self, dtype, *args, **kwargs):
        new_args= []
        new_kwargs={}

        for arg in args:
            if torch.is_tensor(arg):    
                if arg.dtype == torch.float32:
                    arg = arg.to(dtype).cuda(non_blocking=True)
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)
            new_args.append(arg)
        for k in kwargs:
            arg = kwargs[k]
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(dtype).cuda(non_blocking=True)             
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)             
            new_kwargs[k]= arg
        
        return new_args, new_kwargs

    def ready_to_check_mem(self):
        if self.anyCompiledModule:
             return
        cur_clock = time.time()
        # can't check at each call if we can empty the cuda cache as quering the reserved memory value is a time consuming operation
        if (cur_clock - self.last_reserved_mem_check)<0.200:
            return False
        self.last_reserved_mem_check = cur_clock
        return True        


    def empty_cache_if_needed(self):
        mem_reserved = torch.cuda.memory_reserved()
        mem_threshold = 0.9*self.device_mem_capacity
        if mem_reserved >= mem_threshold:            
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved: 
                # print(f"Cuda empty cache triggered as Allocated Memory ({mem_allocated/1024000:0f} MB) is lot less than Cached Memory ({mem_reserved/1024000:0f} MB)  ")
                torch.cuda.empty_cache()
                tm= time.time()
                if self.verboseLevel >=2:
                    print(f"Empty Cuda cache at {tm}")
                # print(f"New cached memory after purge is {torch.cuda.memory_reserved()/1024000:0f} MB)  ")


    def any_param_or_buffer(self, target_module: torch.nn.Module):
        
        for _ in target_module.parameters(recurse= False):
            return True
        
        for _ in target_module.buffers(recurse= False):
            return True
        
        return False

    def _get_lora_scaling(self, loras_scaling, model, active_adapter):
        scaling_list = loras_scaling[active_adapter]
        if isinstance(scaling_list, list):
            step_no =getattr(model, "_lora_step_no", 0)
            return scaling_list[step_no]
        else:
            return float(scaling_list)



    def _lora_generic_forward(self, model, submodule, loras_data, func, *args, **kwargs) -> torch.Tensor:

        weight = submodule.weight 
        bias =  getattr(submodule, "bias", None) 
        original_weight = None 
        original_bias = None
        active_adapters = model._loras_active_adapters
        loras_scaling = model._loras_scaling
        first_weight =  True
        first_bias =  True
        for active_adapter in active_adapters:
            data = loras_data.get(active_adapter + '_GPU', None)
            if data == None:
                continue
            diff_w , _ , diff_b, _, alpha = data
            scaling = self._get_lora_scaling( loras_scaling, model, active_adapter) * alpha
            if scaling == 0:
                continue
            if first_weight:
                original_weight= weight.clone() if weight is not None else None
                first_weight = False
            if first_bias:
                original_bias= bias.clone() if bias is not None else None
                first_bias = False

            if diff_w is not None:
                weight.add_(diff_w, alpha= scaling)
                diff_w = None
            if diff_b is not None:
                bias.add_(diff_b, alpha= scaling)
                diff_b = None

        ret = func(*args, **kwargs )

        if original_weight is not None: weight.data  = original_weight    
        if original_bias is not None: bias.data = original_bias

        return ret


    def _dora_linear_forward(
        self,
        model,
        submodule,
        adapters_data,                # dict: name+"_GPU" -> (A, B, diff_b, g_abs, alpha); g_abs=None means LoRA
        weight= None,
        bias = None,
        original_bias = True,
        dora_mode: str = "blend",     # "ref_exact" | "blend"
    ):
        active_adapters = getattr(model, "_loras_active_adapters", [])
        loras_scaling   = getattr(model, "_loras_scaling", {})
        # Snapshot base weight (safe for quantized modules)
        if weight is None:
            bias = submodule.bias
            original_bias = True
            if isinstance(submodule, QModuleMixin):
                weight = submodule.weight.view(submodule.weight.shape)
            else:
                weight = submodule.weight.clone()

        base_dtype = weight.dtype
        eps = 1e-8
        W0 = weight.float()
        g0 = torch.linalg.vector_norm(W0, dim=1, keepdim=True, dtype=torch.float32).clamp_min(eps)  # [out,1]

        # Keep big mats in low precision
        # Wc = W0 if W0.dtype == compute_dtype else W0.to(compute_dtype)
        W0 /= g0
        weight[...]  = W0.to(base_dtype) 
        W0 = None

        dir_update = None          # Σ s * ((B@A)/g0)  in compute_dtype
        g = None                   # final magnitude: set absolute (ref_exact) or blended (blend)
        bias_delta = None          # Σ s * diff_b

        # Accumulate DoRA adapters only (g_abs != None)
        for name in active_adapters:
            data = adapters_data.get(name + "_GPU", None)
            if data is None: continue
            A, B, diff_b, g_abs, alpha = data
            if g_abs is None: continue  

            s = self._get_lora_scaling(loras_scaling, model, name) * float(alpha)
            if s == 0: continue

            # Direction update in V-space with row-wise 1/g0
            if (A is not None) and (B is not None):
                dV = torch.mm(B, A)      # [out,in], compute_dtype
                dV /= g0               # row-wise divide
                dV.mul_(s)
                dir_update = dV if dir_update is None else dir_update.add_(dV)


            if dora_mode == "ref_exact":
                # absolute magnitude (last one wins if multiple DoRAs present)
                g = g_abs
            elif dora_mode == "blend":
                # blend towards absolute magnitude proportional to s
                if g is None:
                    g = g0.clone()
                g.add_(g_abs.sub(g0), alpha=s)
            else:
                raise ValueError(f"Unknown dora_mode: {dora_mode}")

            # Optional bias deltas (not in reference, but harmless if present)
            if diff_b is not None:
                db = diff_b.mul(s)
                bias_delta = db if bias_delta is None else bias_delta.add_(db)
                db = None

        if g is None:
            g = g0  # no magnitude provided -> keep original

        # Re-normalize rows if we changed direction
        if dir_update is not None:
            weight.add_(dir_update)
            V = weight.float()
            Vn = torch.linalg.vector_norm(V, dim=1, keepdim=True, dtype=torch.float32).clamp_min(eps)
            V /= Vn
            V *= g
            weight[...] = V.to(base_dtype)
            V = None
        else:
            weight *= g
        # Recompose adapted weight; cast back to module dtype

        # Merge DoRA bias delta safely
        if bias_delta is not None:
            if bias is None:
                bias = bias_delta 
            else:
                bias = bias.clone() if original_bias else bias
                bias.add_(bias_delta)

        return weight, bias



    def _lora_linear_forward(self, model, submodule, loras_data, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = submodule.weight
        bias = submodule.bias
        active_adapters = model._loras_active_adapters
        loras_scaling = model._loras_scaling
        any_dora = loras_data.get("any_dora", False)
        is_nvfp4 = getattr(submodule, "is_nvfp4", False)
        training = False

        dtype = weight.dtype
        input_dtype = x.dtype
        if (weight.shape[-1] < x.shape[-2] and False or any_dora): # sum base weight and lora matrices instead of applying input on each sub lora matrice if input is too large. This will save a lot VRAM and compute
            original_bias = True
            original_bias = True
            if len(active_adapters) > 0:
                if isinstance(submodule, QModuleMixin): 
                    weight = weight.view(weight.shape) # get a persistent copy of the on the fly dequantized weights
                else:
                    weight = weight.clone()
                for active_adapter in active_adapters:
                    data = loras_data.get(active_adapter + '_GPU', None)
                    if data == None:
                        continue                    
                    lora_A_weight, lora_B_weight, diff_b, g_abs, alpha = data
                    scaling = self._get_lora_scaling(loras_scaling, model, active_adapter) * alpha
                    if scaling == 0 or g_abs is not None:
                        continue
                    target_dtype = weight.dtype
                    if lora_A_weight is not None and lora_A_weight.dtype != target_dtype:
                        lora_A_weight = lora_A_weight.to(target_dtype)
                    if lora_B_weight is not None and lora_B_weight.dtype != target_dtype:
                        lora_B_weight = lora_B_weight.to(target_dtype)
                    if diff_b is not None and diff_b.dtype != target_dtype:
                        diff_b = diff_b.to(target_dtype)
                    if lora_A_weight != None:
                        weight.addmm_(lora_B_weight, lora_A_weight, alpha= scaling )
                    
                    if diff_b != None:
                        if bias == None:
                            bias = diff_b.clone()
                            original_bias = False
                        elif original_bias:
                            bias = bias.clone()
                            original_bias = False
                        bias.add_(diff_b, alpha=scaling)
                    # base_weight += scaling * lora_B_weight @ lora_A_weight

                if any_dora :
                    weight, bias = self._dora_linear_forward(model, submodule, loras_data, weight, bias, original_bias)

            if training:
                pass
                # result = torch.nn.functional.linear(dropout(x), base_weight, bias=submodule.bias)
            else:
                base_bias = bias
                if base_bias is not None and base_bias.dtype != x.dtype:
                    base_bias = base_bias.to(x.dtype)
                result = torch.nn.functional.linear(x, weight, bias=base_bias)

        else:
            base_bias = bias
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            result = torch.nn.functional.linear(x, weight, bias=base_bias)

            if len(active_adapters) > 0:
                compute_dtype = torch.float32 if is_nvfp4 else result.dtype
                if result.dtype != compute_dtype:
                    result = result.to(compute_dtype)
                x = x.to(compute_dtype)

                for active_adapter in active_adapters:
                    data = loras_data.get(active_adapter + '_GPU', None)
                    if data == None:
                        continue
                    lora_A, lora_B, diff_b, g_abs, alpha = data
                    # dropout = self.lora_dropout[active_adapter]
                    scaling = self._get_lora_scaling(loras_scaling, model, active_adapter) * alpha
                    if scaling == 0 or g_abs is not None:
                        continue
                    target_dtype = result.dtype
                    if lora_A is not None and lora_A.dtype != target_dtype:
                        lora_A = lora_A.to(target_dtype)
                    if lora_B is not None and lora_B.dtype != target_dtype:
                        lora_B = lora_B.to(target_dtype)
                    if diff_b is not None and diff_b.dtype != target_dtype:
                        diff_b = diff_b.to(target_dtype)

                    if lora_A == None:
                        result.add_(diff_b, alpha=scaling)
                    else:
                        x_2d = x.reshape(-1, x.shape[-1])
                        result_2d = result.reshape(-1, result.shape[-1])
                        y = x_2d @ lora_A.T
                        result_2d.addmm_(y, lora_B.T, beta=1, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        del y
                target_dtype = input_dtype if is_nvfp4 else dtype
                if result.dtype != target_dtype:
                    result = result.to(target_dtype)

        return result


    def hook_lora(self, submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name):
        old_forward = submodule.forward

        loras_data = {}
        assert submodule_name not in loras_model_shortcuts 
        loras_model_shortcuts[submodule_name] = loras_data
        loras_model_data[submodule] = loras_data

        if isinstance(submodule,  torch.nn.Linear) or getattr(submodule, "is_nvfp4", False):
            def lora_linear_forward(module,  *args, **kwargs):
                if len(loras_data) == 0:
                    return old_forward(*args, **kwargs)
                else:
                    #submodule.aaa = submodule_name # just for debugging if uncommented will cause pytorch recompilation
                    return self._lora_linear_forward(current_model, submodule, loras_data,  *args, **kwargs)
            target_fn = lora_linear_forward
        else:
            def lora_generic_forward(module,  *args, **kwargs):
                if len(loras_data) == 0:
                    return old_forward(*args, **kwargs)
                else:
                    return self._lora_generic_forward(current_model, submodule, loras_data, old_forward, *args, **kwargs)
            target_fn = lora_generic_forward
        return functools.update_wrapper(functools.partial(target_fn, submodule), old_forward)

    def ensure_model_loaded(self, model_id):
        if model_id in self.active_models_ids:
            return
        # new_model_id = getattr(module, "_mm_id") 
        # do not always unload existing models if it is more efficient to keep in them in the GPU 
        # (e.g: small modules whose calls are text encoders) 
        if not self.can_model_be_cotenant(model_id) :
            self.unload_all()
        self.gpu_load(model_id)

    def hook_preload_blocks_for_compilation(self, target_module, model_id,blocks_name, context):

        # @torch.compiler.disable()
        def preload_blocks_for_compile(module,  *args, **kwargs):
            # some_context = context #for debugging
            if blocks_name != None and blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_load_blocks(model_id, blocks_name)

        # need to be registered before the forward not to be break the efficiency of the compilation chain
        # it should be at the top of the compilation as this type of hook in the middle of a chain seems to break memory performance
        target_module.register_forward_pre_hook(preload_blocks_for_compile)




    @torch._dynamo.disable
    def _pre_check(self, module):
        model_id    = getattr(module, "_mm_model_id", None)
        blocks_name = getattr(module, "_mm_blocks_name", None)

        self.ensure_model_loaded(model_id)
        if blocks_name is None:
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()
        elif blocks_name != self.loaded_blocks[model_id] and \
             blocks_name not in self.preloaded_blocks_per_model[model_id]:
            self.gpu_load_blocks(model_id, blocks_name)

    def _get_wrapper_for_type(self, mod_cls):
        fn = self._type_wrappers.get(mod_cls)
        if fn is not None:
            return fn

        # Unique function name per class -> unique compiled code object
        fname = f"_mm_wrap_{mod_cls.__module__.replace('.', '_')}_{mod_cls.__name__}"

        # Keep body minimal; all heavy/offload logic runs out-of-graph in _pre_check
        # Include __TYPE_CONST in the code so the bytecode/consts differ per class.
        src = f"""
def {fname}(module, *args, **kwargs):
    _ = __TYPE_CONST  # anchor type as a constant to make code object unique per class
    nada = "{fname}"
    mgr = module._mm_manager
    mgr._pre_check(module)
    return module._mm_forward(*args, **kwargs) #{fname}
"""
        ns = {"__TYPE_CONST": mod_cls}
        exec(src, ns)                   # compile a new function object/code object for this class
        fn = ns[fname]
        self._type_wrappers[mod_cls] = fn
        return fn

    def hook_check_load_into_GPU_if_needed(
        self, target_module, model, model_id, blocks_name, previous_method, context
    ):
        # store instance data on the module (not captured by the wrapper)
        target_module._mm_manager     = self
        target_module._mm_model_id    = model_id
        target_module._mm_blocks_name = blocks_name
        target_module._mm_forward     = previous_method

        # per-TYPE wrapper (unique bytecode per class, reused across instances of that class)
        wrapper_fn = self._get_wrapper_for_type(type(target_module))

        # bind as a bound method (no partial/closures)
        # target_module.forward = types.MethodType(wrapper_fn, target_module)
        target_module.forward = functools.update_wrapper(functools.partial(wrapper_fn, target_module), previous_method) 

    def hook_check_load_into_GPU_if_needed_default(self, target_module, model, model_id, blocks_name, previous_method,  context):

        dtype = model._dtype
        qint4quantization =  isinstance(target_module, QModuleMixin) and  target_module.weight!= None and  target_module.weight.qtype == qint4 
        if qint4quantization:
            pass

        if hasattr(target_module, "_mm_id"):
            # no hook for a shared module with no weights (otherwise this will cause models loading / unloading for nothing)
            orig_model_id = getattr(target_module, "_mm_id")
            if self.verboseLevel >=2:
                print(f"Model '{model_id}' shares module '{target_module._get_name()}' with module(s) '{orig_model_id}' ")
            assert not self.any_param_or_buffer(target_module)
            if not isinstance(orig_model_id, list):
                orig_model_id = [orig_model_id]
            orig_model_id.append(model_id)
            setattr(target_module, "_mm_id", orig_model_id)
            target_module.forward = target_module._mm_forward
            return

        def check_load_into_GPU_needed():
            self.ensure_model_loaded(model_id)
            if blocks_name == None:
                if self.ready_to_check_mem():
                    self.empty_cache_if_needed()
            elif blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_load_blocks(model_id, blocks_name)
            # if qint4quantization and dtype !=None:
            #     args, kwargs = self.move_args_to_gpu(dtype, *args, **kwargs)

        if isinstance(target_module, torch.nn.Linear):
            def check_load_into_GPU_needed_linear(module, *args, **kwargs):
                check_load_into_GPU_needed()
                return previous_method(*args, **kwargs) # linear
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_linear
        else:
            def check_load_into_GPU_needed_other(module, *args, **kwargs):
                check_load_into_GPU_needed()
                return previous_method(*args, **kwargs) # other
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_other

        setattr(target_module, "_mm_id", model_id)
        setattr(target_module, "_mm_forward", previous_method)

        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_load_into_GPU_needed_module, target_module), previous_method) )
        # target_module.register_forward_pre_hook(check_empty_cuda_cache)

        
    def hook_change_module(self, target_module, model, model_id, module_id, previous_method, previous_method_name ):
        if hasattr(target_module, "_lock_dtype"):
            dtype = target_module._lock_dtype 
        else:
            dtype = model._dtype

        def check_change_module(module, *args, **kwargs):      
            self.ensure_model_loaded(model_id)
            # transfer leftovers inputs that were incorrectly created in the RAM (mostly due to some .device tests that returned incorrectly "cpu")
            if dtype != None:
                args, kwargs = self.move_args_to_gpu(dtype, *args, **kwargs)
            return previous_method(*args, **kwargs) 
  
        if hasattr(target_module, "_mm_" + previous_method_name):
            return
        setattr(target_module, "_mm_Id", model_id)
        setattr(target_module, "_mm_" + previous_method_name, previous_method)

        setattr(target_module, previous_method_name, functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method) )

        if not self.verboseLevel >=1:
            return

        if previous_method_name =="forward" and (module_id == None or module_id ==''):
            model_name = model._get_name()
            print(f"Hooked to model '{model_id}' ({model_name})")



    def tune_preloading(self, model_id, current_budget, towers_names):
        preloaded_blocks = {}
        preload_total = 0
        max_blocks_fetch = 0

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks

        if current_budget == 0 or towers_names is None or len(towers_names) == 0 or not self.async_transfers:
            return
        base_size = self.blocks_of_modules_sizes[model_id] 
        current_budget -= base_size
        current_budget = max(0, current_budget)
        
        towers = []
        total_size = 0
        for tower_name in towers_names:
            max_floor_size = 0
            tower_size = 0
            floors = []
            prefix = model_id + "/" + tower_name
            for name, size in self.blocks_of_modules_sizes.items():
                if name.startswith(prefix):
                    tower_size += size
                    floor_no = int(  name[len(prefix): ] )
                    floors.append( (name, floor_no, size))
                    max_floor_size = max(max_floor_size, size)

            towers.append( (floors, max_floor_size, tower_size) )
            total_size += tower_size
            current_budget -=  2 * max_floor_size
            current_budget = max(0, current_budget)

        for floors, max_floor_size, tower_size in towers:
            tower_budget = tower_size / total_size * current_budget
            preload_blocks_count = int( tower_budget / max_floor_size)
            preload_total += preload_blocks_count * max_floor_size
            max_blocks_fetch = max(max_floor_size, max_blocks_fetch)
            
            nb_blocks= len(floors)
            if preload_blocks_count == 0:
                space_between = 0
                cursor = len(floors)
            else:
                space_between =  (nb_blocks - preload_blocks_count) / preload_blocks_count 
                cursor = space_between
            first_non_preloaded = None
            prev_non_preloaded = None
            for block in floors:
                name, i, size = block
                if i < cursor:
                    if prev_non_preloaded == None:
                        first_non_preloaded = name
                    else:
                        self.next_blocks_names[prev_non_preloaded] = name
                        self.prev_blocks_names[name] = prev_non_preloaded
                    prev_non_preloaded = name
                else:
                    self.next_blocks_names[name] = None
                    self.prev_blocks_names[name] = None
                    preloaded_blocks[name[ len(model_id) + 1 : ] ] = size
                    cursor += 1 + space_between

            if prev_non_preloaded != None and len(towers) == 1 : 
                self.next_blocks_names[prev_non_preloaded] = first_non_preloaded
                self.prev_blocks_names[first_non_preloaded] = prev_non_preloaded
            else:
                self.next_blocks_names[prev_non_preloaded] = None

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks

        if self.verboseLevel >=1:
            if preload_total == 0:
                print(f"Async loading plan for model '{model_id}' : base size of {(preload_total+base_size)/ONE_MB:0.2f} MB will be preloaded with a {max_blocks_fetch/ONE_MB:0.2f} MB async" + (" circular" if len(towers) == 1 else "") + " shuttle")
            else:
                print(f"Async loading plan for model '{model_id}' : {(preload_total+base_size)/ONE_MB:0.2f} MB will be preloaded (base size of {base_size/ONE_MB:0.2f} MB + {preload_total/total_size*100:0.1f}% of recurrent layers data) with a {max_blocks_fetch/ONE_MB:0.2f} MB async" + (" circular" if len(towers) == 1 else "") + " shuttle")

    def release(self):
        global last_offload_obj, total_pinned_bytes

        if last_offload_obj == self:
            last_offload_obj = None

        self.unload_all()
        self.active_models = None
        self.default_stream = None 
        self.transfer_stream = None
        self.parameters_ref = None
        keys= [k for k in self.blocks_of_modules.keys()]
        for k in keys:
            del self.blocks_of_modules[k]

        self.blocks_of_modules = None

        for model_id, model in self.models.items():
            move_loras_to_device(model, "cpu")
            if hasattr(model, "_pinned_bytes"):
                total_pinned_bytes -= model._pinned_bytes
            if hasattr(model, "_loras_model_data"):
                unload_loras_from_model(model)
            model = None

        self.models = None            

        gc.collect()
        torch.cuda.empty_cache()




def all(pipe_or_dict_of_modules, pinnedMemory = False, pinnedPEFTLora = False, partialPinning = False, loras = None, quantizeTransformer = True,  extraModelsToQuantize = None, quantizationType = qint8, budgets= 0, workingVRAM = None, asyncTransfers = True, compile = False, convertWeightsFloatTo = torch.bfloat16, perc_reserved_mem_max = 0, coTenantsMap = None, vram_safety_coefficient = 0.8, compile_mode ="default", verboseLevel = -1):
    """Hook to a pipeline or a group of modules in order to reduce their VRAM requirements:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    quantizeTransformer: set True by default will quantize on the fly the video / image model
    pinnedMemory: move models in reserved memor. This allows very fast performance but requires 50% extra RAM (usually >=64 GB)
    extraModelsToQuantize: a list of models to be also quantized on the fly (e.g the text_encoder), useful to reduce bith RAM and VRAM consumption
    budgets: 0 by default (unlimited). If non 0, it corresponds to the maximum size in MB that every model will occupy at any moment
        (in fact the real usage is twice this number). It is very efficient to reduce VRAM consumption but this feature may be very slow
        if pinnedMemory is not enabled
    vram_safety_coefficient: float between 0 and 1 (exclusive), default 0.8. Sets the maximum portion of VRAM that can be used for models.
        Lower values provide more safety margin but may reduce performance.        
    """
    self = offload()
    self.verboseLevel = verboseLevel
    safetensors2.verboseLevel = verboseLevel
    self.modules_data = {}
    
    model_budgets = {}

    windows_os =  os.name == 'nt'

    def get_parsed_budget(b):
        if isinstance(b , str) and b.endswith("%"):
            return float(b[:-1]) * self.device_mem_capacity
        else:
            return b * ONE_MB

    # Validate vram_safety_coefficient
    if not isinstance(vram_safety_coefficient, float) or vram_safety_coefficient <= 0 or vram_safety_coefficient >= 1:
        raise ValueError("vram_safety_coefficient must be a float between 0 and 1 (exclusive)")

    budget = 0
    if not budgets is None:
        if isinstance(budgets , dict):
            model_budgets = { k : get_parsed_budget(b) for k , b in budgets.items() } 
            budget = model_budgets.get("*", 0)
        else:
            budget = get_parsed_budget(budget) 

    self.async_transfers = asyncTransfers



    torch.set_default_device('cpu')

    if hasattr(pipe_or_dict_of_modules, "components"):
        # create a fake Accelerate parameter so that lora loading doesn't change the device
        pipe_or_dict_of_modules.hf_device_map = torch.device("cuda")
        pipe_or_dict_of_modules= pipe_or_dict_of_modules.components 

    
    models = {k: _remove_model_wrapper(v) for k, v in pipe_or_dict_of_modules.items() if isinstance(v, torch.nn.Module)}

    
    verboseLevel = _compute_verbose_level(verboseLevel)

    _welcome()        
    if coTenantsMap != None:
        self.cotenants_map = coTenantsMap 
    if loras != None and isinstance(loras, str):
        loras = [loras]
    self.models = models

    extraModelsToQuantize =  extraModelsToQuantize if extraModelsToQuantize is not None else []
    if not isinstance(extraModelsToQuantize, list):
        extraModelsToQuantize= [extraModelsToQuantize]
    if quantizeTransformer:
        extraModelsToQuantize.append("transformer")            
    models_to_quantize = extraModelsToQuantize

    modelsToPin = []
    pinAllModels = False
    if isinstance(pinnedMemory, bool):
        pinAllModels = pinnedMemory
    elif isinstance(pinnedMemory, list):            
        modelsToPin = pinnedMemory
    else:
        modelsToPin = [pinnedMemory]

    modelsToCompile = []
    compileAllModels = False
    if isinstance(compile, bool):
        compileAllModels = compile
    elif isinstance(compile, list):            
        modelsToCompile = compile
    else:
        modelsToCompile = [compile]

    self.anyCompiledModule = compileAllModels or len(modelsToCompile)>0
    if self.anyCompiledModule:
        torch.compiler.reset()
        torch._dynamo.config.cache_size_limit = 10000
    #dynamic=True

      #  torch._logging.set_logs(recompiles=True)
      #  torch._inductor.config.realize_opcount_threshold = 100 # workaround bug "AssertionError: increase TRITON_MAX_BLOCK['X'] to 4096."

    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max)
    max_reservable_memory = _get_max_reservable_memory(perc_reserved_mem_max) 

    estimatesBytesToPin = 0
    for model_id in models: 
        current_model: torch.nn.Module = models[model_id] 
        # make sure that no RAM or GPU memory is not allocated for gradiant / training
        current_model.to("cpu").eval()
        
        # if the model has just been quantized so there is no need to quantize it again
        if model_id in models_to_quantize:
            _quantize(current_model, weights=quantizationType, verboseLevel = self.verboseLevel, model_id=model_id)

        modelPinned = (pinAllModels or model_id in modelsToPin) and not hasattr(current_model,"_already_pinned")

        current_model_size = 0
        model_dtype = getattr(current_model, "_model_dtype", None)
        # if model_dtype == None:
        #     model_dtype = getattr(current_model, "dtype", None)
        for _ , m in current_model.named_modules():
            ignore_dtype = hasattr(m, "_lock_dtype")
            for n, p in m.named_parameters(recurse = False):
                p.requires_grad = False
                sub_tensors = _get_quantized_subtensors(p)
                if sub_tensors:
                    current_model_size += _subtensors_nbytes(sub_tensors)
                    dtype = sub_tensors[0][1].dtype
                    for name, tensor in sub_tensors:
                        if name in ("scale", "scale_shift"):
                            dtype = tensor.dtype
                            break
                    del sub_tensors
                else:
                    if not ignore_dtype:
                        dtype = p.data.dtype
                        if convertWeightsFloatTo != None and dtype == torch.float32 :
                            # convert any left overs float32 weight to bfloat16 / float16 to divide by 2 the model memory footprint
                            dtype = convertWeightsFloatTo if model_dtype == None else model_dtype
                            if dtype != torch.float32:
                                p.data = p.data.to(dtype)
                        if model_dtype== None:
                            model_dtype = dtype
                        else:
                            if model_dtype != dtype:
                                pass
                            assert model_dtype == dtype
                    current_model_size +=  torch.numel(p.data) * p.data.element_size()
                current_model._dtype = model_dtype
        for b in current_model.buffers():
            # do not convert 32 bits float to 16 bits since buffers are few (and potential gain low) and usually they are needed for precision calculation (for instance Rope)
            current_model_size +=  torch.numel(b.data) * b.data.element_size()

        if modelPinned:
            estimatesBytesToPin += current_model_size
        

        model_budget = model_budgets[model_id] if model_id in model_budgets else budget
        if workingVRAM != None:
            model_minimumVRAM = -1
            if isinstance(workingVRAM, dict):
                if model_id in workingVRAM:
                    model_minimumVRAM = get_parsed_budget(workingVRAM[model_id])
                elif "*" in model_id in workingVRAM:
                    model_minimumVRAM = get_parsed_budget(workingVRAM["*"])
            else:
                model_minimumVRAM = get_parsed_budget(workingVRAM)

            if model_minimumVRAM > 0:
                new_budget = self.device_mem_capacity -  model_minimumVRAM
                new_budget = 1 if new_budget  < 0 else new_budget
                model_budget =  new_budget if model_budget == 0 or new_budget < model_budget else model_budget
        if  model_budget > 0 and model_budget > current_model_size:
            model_budget = 0
        coef =vram_safety_coefficient
        if current_model_size > coef * self.device_mem_capacity and model_budget == 0 or model_budget > coef * self.device_mem_capacity:
            if verboseLevel >= 1:
                if model_budget == 0:
                    print(f"Model '{model_id}' is too large ({current_model_size/ONE_MB:0.1f} MB) to fit entirely in {coef * 100:.0f}% of the VRAM (max capacity is {coef * self.device_mem_capacity/ONE_MB:0.1f}) MB)")
                else:
                    print(f"Budget ({budget/ONE_MB:0.1f} MB) for Model '{model_id}' is too important so that this model can fit in the VRAM (max capacity is {self.device_mem_capacity/ONE_MB}) MB)")
                print(f"Budget allocation for this model has been consequently reduced to the {coef * 100:.0f}% of max GPU Memory ({coef * self.device_mem_capacity/ONE_MB:0.1f} MB). This may not leave enough working VRAM and you will probably need to define manually a lower budget for this model.")
                model_budget = coef * self.device_mem_capacity 
                
        
        model_budgets[model_id] = model_budget


    if not partialPinning and estimatesBytesToPin > 0 and estimatesBytesToPin >= (max_reservable_memory - total_pinned_bytes):
        if self.verboseLevel >=1:
            print(f"Switching to partial pinning since full requirements for pinned models is {estimatesBytesToPin/ONE_MB:0.1f} MB while estimated available reservable RAM is {(max_reservable_memory-total_pinned_bytes)/ONE_MB:0.1f} MB. You may increase the value of parameter 'perc_reserved_mem_max' to a value higher than {perc_reserved_mem_max:0.2f} to force full pinnning." )
        partialPinning = True

    #  Hook forward methods of modules 
    for model_id in models: 
        current_model: torch.nn.Module = models[model_id] 
        towers_names, towers_modules = _detect_main_towers(current_model)
        compilationInThisOne = compileAllModels or model_id in modelsToCompile 
                
        if pinAllModels or model_id in modelsToPin:
            if hasattr(current_model,"_already_pinned"):
                if self.verboseLevel >=1:
                    print(f"Model '{model_id}' already pinned to reserved memory")
            else:
                _pin_to_memory(current_model, model_id, partialPinning= partialPinning, pinnedPEFTLora = pinnedPEFTLora, perc_reserved_mem_max = perc_reserved_mem_max, verboseLevel=verboseLevel)            
                # empty_tensor = torch.empty((1,))
                # for sub_module_name, sub_module  in current_model.named_modules():
                #     for k, p in  sub_module.named_parameters(recurse=False):
                #         if p is not None:
                #             if isinstance(p, QTensor):
                #                 p._data.data = empty_tensor
                #                 p._scale.data = empty_tensor
                #             else:
                #                 p.data = empty_tensor
                #             del k
                #     for k, v in  sub_module.named_buffers(recurse=False):
                #         del k
                # sub_module = None
                # v = None
                # gc.collect()
        current_budget = model_budgets[model_id]
        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False
        self.loaded_blocks[model_id] = None
        any_lora =  loras !=None and model_id in loras
        if any_lora: 
            loras_model_data, loras_model_shortcuts = {}, {}
            current_model._loras_model_data = loras_model_data 
            current_model._loras_model_shortcuts = loras_model_shortcuts
        for submodule_name, submodule in current_model.named_modules():  
            # create a fake 'accelerate' parameter so that the _execution_device property returns always "cuda" 
            # (it is queried in many pipelines even if offloading is not properly implemented)  
            if not hasattr(submodule, "_hf_hook"):
                setattr(submodule, "_hf_hook", HfHook())
            if current_budget > 0 and len(submodule_name) > 0:
                if cur_blocks_prefix != None:
                    if submodule_name.startswith(cur_blocks_prefix):
                        depth_prefix = cur_blocks_prefix.split(".")
                        depth_name = submodule_name.split(".")
                        level  =  depth_name[len(depth_prefix)-1]                        
                        pre , num = _extract_num_from_str(level)
                        if num != cur_blocks_seq and not (is_mod_seq and cur_blocks_seq>=0):
                            prev_blocks_name = cur_blocks_name
                            cur_blocks_name =  cur_blocks_prefix + str(num)
                            # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
                        cur_blocks_seq = num
                    else:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False

                if cur_blocks_prefix == None:
                    pre , num = _extract_num_from_str(submodule_name)
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):  
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre + ".", None, -1, isinstance(submodule, torch.nn.Sequential)
                    elif num >=0:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre, None, num, False
                        cur_blocks_name = submodule_name
                        # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
            top_submodule = len(submodule_name.split("."))==1
            offload_hooks = submodule._offload_hooks if hasattr(submodule, "_offload_hooks") else [] 
            assert top_submodule or len(offload_hooks) == 0, "custom offload hooks can only be set at the of the module"
            submodule_method_names = ["forward"] +  offload_hooks
            for submodule_method_name in submodule_method_names:
                if not hasattr(submodule, submodule_method_name ): continue
                if submodule_method_name == "forward" and any_lora and hasattr(submodule,"weight"):
                    submodule_method = self.hook_lora(submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name)                
                else:
                    submodule_method = getattr(submodule, submodule_method_name)
                if callable(submodule_method):
                    if top_submodule and cur_blocks_name is None:
                        self.hook_change_module(submodule, current_model, model_id, submodule_name, submodule_method, submodule_method_name)
                    elif compilationInThisOne and submodule in towers_modules: 
                        self.hook_preload_blocks_for_compilation(submodule, model_id, cur_blocks_name, context = submodule_name )
                    else:
                        if compilationInThisOne: #and False
                            self.hook_check_load_into_GPU_if_needed(submodule, current_model, model_id, cur_blocks_name, submodule_method, context = submodule_name )
                        else:
                            self.hook_check_load_into_GPU_if_needed_default(submodule, current_model, model_id, cur_blocks_name, submodule_method, context = submodule_name )

                    self.add_module_to_blocks(model_id, cur_blocks_name, submodule, prev_blocks_name, submodule_name)


        # compile main iterative modules stacks ("towers")
        if compilationInThisOne:
            if self.verboseLevel>=1:
                if len(towers_modules)>0:
                    formated_tower_names = [name + '*' for name in towers_names]
                    print(f"Pytorch compilation of '{model_id}' is scheduled for these modules : {formated_tower_names}.")
                else:
                    print(f"Pytorch compilation of model '{model_id}' is not yet supported.")

            for submodel in towers_modules:
                submodel.forward= torch.compile(submodel.forward,  backend= "inductor", mode= compile_mode) # , fullgraph= True, mode= "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",  
                    #dynamic=True,

        self.tune_preloading(model_id, current_budget, towers_names)
        self.parameters_ref  = {} 


    if self.verboseLevel >=2:
        start_num, prev_num, prev_pre, prev_size  = -1, -1, None, -1
         
        def print_size_range(n,start_num,prev_num, prev_size ):
            if prev_num < 0:
                print(f"Size of submodel '{n}': {prev_size/ONE_MB:.1f} MB")
            elif prev_num - start_num <=1:
                print(f"Size of submodel '{n+ str(start_num)}': {prev_size/ONE_MB:.1f} MB")
            else:
                print(f"Size of submodel '{n+ str(start_num) +'-'+ str(prev_num)}': {(prev_num-start_num+1)*prev_size/ONE_MB:.1f} MB ({prev_size/ONE_MB:.1f} MB x {prev_num-start_num+1})")

        for n, size in self.blocks_of_modules_sizes.items():
            size = int(size / 10000)* 10000
            pre, num = _extract_num_from_str(n) if "/" in n else (n, -1)
            if prev_pre == None :
                start_num = num
            elif prev_pre != pre or prev_pre == pre and size != prev_size:
                print_size_range(prev_pre,start_num,prev_num, prev_size )
                start_num = num
            prev_num, prev_pre, prev_size = num, pre, size
        if prev_pre != None:
            print_size_range(prev_pre,start_num,prev_num, prev_size )

  
    torch.set_default_device('cuda')
    torch.cuda.empty_cache()
    gc.collect()         

    return self


def profile(pipe_or_dict_of_modules, profile_no: profile_type =  profile_type.VerylowRAM_LowVRAM, verboseLevel = -1, **overrideKwargs):
    """Apply a configuration profile that depends on your hardware:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    profile_name : num of the profile:
        HighRAM_HighVRAM_Fastest (=1): will try to load entirely a model  in VRAM and to keep a copy in reserved RAM for fast loading / unloading
        HighRAM_LowVRAM_Fast (=2): will try to load only the needed parts of a model in VRAM and to keep a copy in reserved RAM for fast loading / unloading
        LowRAM_HighVRAM_Medium (=3): will try to load entirely a model  in VRAM and to keep a copy in reserved RAM for fast loading / unloading, 8 bits quantization of main model
        LowRAM_LowVRAM_Slow (=4): will try to load only the needed parts of a model in VRAM and to keep a copy in reserved RAM for fast loading / unloading, 8 bits quantization of main models
        VerylowRAM_LowVRAM_Slowest (=5): will try to load only the needed parts of a model in VRAM, 8 bits quantization of main models
    overrideKwargs: every parameter accepted by Offload.All can be added here to override the profile choice
        For instance set quantizeTransformer = False to disable transformer quantization which is by default in every profile
    """      

    _welcome()

    verboseLevel = _compute_verbose_level(verboseLevel)

    modules = pipe_or_dict_of_modules

    if hasattr(modules, "components"):
        modules= modules.components 

    modules = {k: _remove_model_wrapper(v) for k, v in modules.items() if isinstance(v, torch.nn.Module)}
    module_names = {k: _get_module_name(v) for k, v in modules.items() }

    default_extraModelsToQuantize = []
    quantizeTransformer = True
    
    models_to_scan = ("text_encoder", "text_encoder_2")
    candidates_to_quantize = ("t5", "llama", "llm")
    for model_id  in models_to_scan:
        if model_id in module_names: 
            name = module_names[model_id]
            for candidate in candidates_to_quantize:
                if candidate in name:
                    default_extraModelsToQuantize.append(model_id)
                    break


    # transformer (video or image generator) should be as small as possible not to occupy space that could be used by actual image data
    # on the other hand the text encoder should be quite large (as long as it fits in 10 GB of VRAM) to reduce sequence offloading

    budgets = {}
    if "transformer" in modules:
        budgets["transformer"] = 1200    

    extraModelsToQuantize = None
    asyncTransfers = True

    if profile_no == profile_type.HighRAM_HighVRAM:
        pinnedMemory= True
        budgets = None
        # info = "You have chosen a profile that may require 48 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.HighRAM_LowVRAM:
        pinnedMemory= True
        budgets["*"] =  3000
        # info = "You have chosen a profile that may require 48 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_HighVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets = None
        # info = "You have chosen a Medium speed profile that may require 32 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_LowVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        # info = "You have chosen a profile that usually may require 32 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.VerylowRAM_LowVRAM:
        pinnedMemory= False
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        if "transformer" in modules:
            budgets["transformer"] = 400    
        #asyncTransfers = False
        # info = "You have chosen the slowest profile that may require 24 GB of RAM and up to 10 GB of VRAM on some applications."
    else:
        raise Exception("Unknown profile")
    # info += " Actual requirements may varry depending on the application or on the tuning done to the profile."
    info =""    
    if budgets != None and len(budgets) == 0:
        budgets = None

    CrLf = '\r\n'
    kwargs = { "pinnedMemory": pinnedMemory,  "extraModelsToQuantize" : extraModelsToQuantize, "budgets": budgets, "asyncTransfers" : asyncTransfers, "quantizeTransformer": quantizeTransformer   }

    if verboseLevel>=2:
        info = info  + f"Profile '{profile_type.tostr(profile_no)}' sets the following options:" #CrLf 
        for k,v in kwargs.items():
            if k in overrideKwargs: 
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}' overriden with value '{overrideKwargs[k]}'"
            else:
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}'"

    for k,v in overrideKwargs.items():
        kwargs[k] = overrideKwargs[k]

    if info:
        print(info)

    return all(pipe_or_dict_of_modules, verboseLevel = verboseLevel, **kwargs)
