# ------------------ Memory Management 3.7.5 for the GPU Poor by DeepBeepMeep (mmgp)------------------
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
import inspect

from mmgp import safetensors2
from mmgp import profile_type
from .quant_router import (
    apply_pre_quantization,
    cache_quantization_for_file,
    detect_and_convert,
    detect_safetensors_format,
    sd_split_linear,
    split_linear_modules,
    split_fused_weights,
    get_extension_handler,
    normalize_extension_path,
)
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

    return _orig_get_parameter_device(parameter)

try:
    import transformers.modeling_utils as mu # Transfomers v4
    _orig_get_parameter_device = mu.get_parameter_device
    def _patched_get_parameter_device(parameter):
        forced = getattr(parameter, "_force_device", None)
        if forced is not None:
            return torch.device(forced)
        return _orig_get_parameter_device(parameter)    
    mu.get_parameter_device = _patched_get_parameter_device	
except:
    pass

try:
    import transformers.modeling_utils as mu # Transfomers v5
    _orig_device = mu.ModuleUtilsMixin.device.fget
    def _device(self):
        forced = getattr(self, "_force_device", None)
        if forced is not None:
            return torch.device(forced)
        return _orig_device(self)    
    mu.ModuleUtilsMixin.device = property(_device)
except:
    pass


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
        # Preserve raw bytes for 0-dim tensors (scalar buffers like embed_scale).
        t = source_tensor.view(1).view(torch.uint8)
        t = torch.reshape(t, (length,))
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
    q = _make_buffer(p.clone())
    torch.utils.swap_tensors(p, q)
    del q

def _force_load_parameter(p):
    q = _make_parameter(p.clone(), requires_grad=p.requires_grad)
    torch.utils.swap_tensors(p, q)
    del q

def _make_buffer(tensor):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Buffer(tensor)
    return torch.nn.Buffer(tensor)

def _make_parameter(tensor, requires_grad=False):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Parameter(tensor, requires_grad=requires_grad)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)

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
    sizes = [t.element_size() for _, t in sub_tensors]
    return max(sizes) if sizes else fallback

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
        
    if verboseLevel >=2 and tied_weights_count > 0:
        if tied_weights_count == 1:
            print(f"Tied weights of {tied_weights_total/ONE_MB:0.2f} MB detected: {tied_weights_last}")
        else:
            print(f"Found {tied_weights_count} tied weights for a total of {tied_weights_total/ONE_MB:0.2f} MB, last : {tied_weights_last}")
    return tied_weights

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
                new_tied_weights[str(i) + "#" + k] = str(i) + "#" + v
        tied_weights = new_tied_weights
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
        dummy_pinned_tensor = None
        flush_torch_caches()
        return
    
    for size in big_tensors_sizes:
        try:
            current_big_tensor = torch.empty( size, dtype= torch.uint8, pin_memory=True, device="cpu")
            big_tensors.append(current_big_tensor)
        except:
            incomplete_pinning = True
            current_big_tensor = None
            print(f"Unable to pin more tensors for '{sd_name}' as the maximum reservable memory has been reached ({total/ONE_MB:.2f}). Transfer speed from RAM to VRAM may be slower.")
            flush_torch_caches()
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
        dummy_pinned_tensor = None
        flush_torch_caches()
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
                    flush_torch_caches()
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
                    tensor = None
                _set_quantized_subtensors(p, new_subs)
                del new_subs, sub_tensors
            else:
                length = torch.numel(p.data) * p.data.element_size()
                p.data = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length)

            tensor_no += 1
        del p
    del dummy_pinned_tensor,tied_weights, ref_cache
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
    print(f"{BOLD}{HEADER}************ Memory Management for the GPU Poor (mmgp 3.7.5) by DeepBeepMeep ************{ENDC}{UNBOLD}")

def change_dtype(model, new_dtype, exclude_buffers = False):
    for submodule_name, submodule in model.named_modules():  
        if hasattr(submodule, "_lock_dtype"):
            continue
        for n, p in submodule.named_parameters(recurse = False):
            if isinstance(p, QTensor):
                continue
            if p.data.dtype != new_dtype:
                p.data = p.data.to(new_dtype)

        if not exclude_buffers:
            for p in submodule.buffers(recurse=False):
                if isinstance(p, QTensor):
                    continue
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
    quantized_names = set(quantization_map.keys())

    def _is_quantized_param(param_name):
        if param_name in quantized_names:
            return True
        if "." in param_name:
            return param_name.rsplit(".", 1)[0] in quantized_names
        return False

    # change dtype of current meta model parameters because 'requantize' won't update the dtype on non quantized parameters
    for k, p in model.named_parameters():
        if _is_quantized_param(k) or k not in state_dict:
            continue
        p_in_file = state_dict[k]
        if not (p_in_file.data.dtype.is_floating_point or p_in_file.data.dtype.is_complex):
            continue
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


    if exclude_list is None:
        exclude_list = list(tied_weights)
    else:
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
def load_loras_into_model(model, lora_path, lora_multi = None, activate_all_loras = True, check_only = False, ignore_model_variations = False, pinnedLora = False, maxReservedLoras = -1, split_linear_modules_map = None, preprocess_sd = None, verboseLevel = -1,):
    verboseLevel = _compute_verbose_level(verboseLevel)

    loras_model_data = getattr(model, "_loras_model_data", None)
    if loras_model_data == None:
        merged_loras_model_data = {}
        merged_loras_shortcuts = {}
        sub_loras = {}
        for submodule_name, submodule in model.named_modules():
            if submodule is model:
                continue
            sub_model_data = getattr(submodule, "_loras_model_data", None)
            if sub_model_data:
                submodule._lora_owner = model
                sub_loras[submodule_name] = submodule
                for k, v in sub_model_data.items():
                    if k not in merged_loras_model_data:
                        merged_loras_model_data[k] = v
            sub_shortcuts = getattr(submodule, "_loras_model_shortcuts", None)
            if sub_shortcuts:
                prefix = f"{submodule_name}." if submodule_name else ""
                for k, v in sub_shortcuts.items():
                    merged_key = k
                    if prefix:
                        if k:
                            merged_key = f"{prefix}{k}"
                        else:
                            merged_key = submodule_name
                    if merged_key not in merged_loras_shortcuts:
                        merged_loras_shortcuts[merged_key] = v
        if merged_loras_model_data:
            model._loras_model_data = merged_loras_model_data
            if merged_loras_shortcuts:
                model._loras_model_shortcuts = merged_loras_shortcuts
            model._subloras = sub_loras
            loras_model_data = merged_loras_model_data
        else:
            raise Exception(f"No Loras has been declared for this model while creating the corresponding offload object")
    
    if not check_only:
        unload_loras_from_model(model)

    modules_dict = {k: v for k,v in model.named_modules()}
    module_names = {v: k for k, v in modules_dict.items()}

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
    
    def _state_dict_size_mb(state_dict):
        total_bytes = 0
        for v in state_dict.values():
            if torch.is_tensor(v):
                total_bytes += v.numel() * v.element_size()
        return total_bytes / (1024 * 1024)

    def _split_lokr_output_sizes(split_sizes, w1, w2):
        if w1 is None or w2 is None:
            return None
        rows2 = int(w2.shape[0])
        if rows2 <= 0:
            return None
        out_rows = int(w1.shape[0]) * rows2
        if sum(split_sizes) != out_rows:
            return None
        reduced = []
        for sz in split_sizes:
            if sz % rows2 != 0:
                return None
            reduced.append(sz // rows2)
        if sum(reduced) != int(w1.shape[0]):
            return None
        return reduced

    def _split_lokr_chunked_if_needed(parent_prefix, mapped_modules, split_sizes, w1, w2, new_state_dict, lokr_split_chunks):
        if w1 is None or w2 is None:
            return False
        out_rows = int(w1.shape[0]) * int(w2.shape[0])
        if sum(split_sizes) != out_rows:
            return False
        start = 0
        for sub_name, sub_size in zip(mapped_modules, split_sizes):
            target_base = parent_prefix + sub_name
            if target_base + ".lokr_w1" not in new_state_dict:
                new_state_dict[target_base + ".lokr_w1"] = w1
            if target_base + ".lokr_w2" not in new_state_dict:
                new_state_dict[target_base + ".lokr_w2"] = w2
            lokr_split_chunks[target_base] = (start, start + sub_size)
            start += sub_size
        return True

    def _split_lokr_key(module_name, module_data, state_dict, new_state_dict, lokr_split_chunks):
        name_parts = module_name.split(".")
        if len(name_parts) < 2:
            return module_name
        split_map = split_linear_modules_map.get(name_parts[-2], None)
        if split_map is None:
            return module_name
        mapped_modules = split_map.get("mapped_modules", None)
        split_sizes = split_map.get("split_sizes", None)
        if not mapped_modules or not split_sizes:
            return module_name

        parent_module_name = ".".join(name_parts[:-1])
        is_w1 = module_name.endswith(".lokr_w1")
        sibling_suffix = ".lokr_w2" if is_w1 else ".lokr_w1"
        sibling_tensor = state_dict.get(parent_module_name + sibling_suffix, None)
        w1 = module_data if is_w1 else sibling_tensor
        w2 = sibling_tensor if is_w1 else module_data
        reduced_split = _split_lokr_output_sizes(split_sizes, w1, w2)
        parent_prefix = parent_module_name[:-len(name_parts[-2])]
        if reduced_split is None:
            if _split_lokr_chunked_if_needed(parent_prefix, mapped_modules, split_sizes, w1, w2, new_state_dict, lokr_split_chunks):
                return None
            return module_name

        if is_w1:
            chunks = torch.split(module_data, reduced_split, dim=0)
            suffix = ".lokr_w1"
        else:
            chunks = [module_data] * len(mapped_modules)
            suffix = ".lokr_w2"
        for sub_name, sub_data in zip(mapped_modules, chunks):
            new_state_dict[parent_prefix + sub_name + suffix] = sub_data
        return None

    def _build_lokr_chunk_specs(chunk, m2):
        start, end = int(chunk[0]), int(chunk[1])
        if end <= start or m2 <= 0:
            return []
        s_row, s_col = divmod(start, m2)
        e_row, e_col = divmod(end, m2)
        if s_row == e_row:
            return [(s_row, s_row + 1, s_col, e_col)]

        specs = []
        mid_start_row = s_row
        if s_col > 0:
            specs.append((s_row, s_row + 1, s_col, m2))
            mid_start_row += 1
        if mid_start_row <= e_row - 1:
            specs.append((mid_start_row, e_row, 0, m2))
        if e_col > 0:
            specs.append((e_row, e_row + 1, 0, e_col))
        return specs

    def _ensure_adapter_meta(loras_module_data, adapter_name):
        data = loras_module_data.get(adapter_name, None)
        if data is None:
            data = [None, None, None, None, 1., {}]
            loras_module_data[adapter_name] = data
        return data, data[5] 

    def _to_dtype_cached(tensor, dtype, cast_cache):
        if tensor is None or tensor.dtype == dtype:
            return tensor
        key = (_get_tensor_ref(tensor), dtype)
        cached = cast_cache.get(key, None)
        if cached is None:
            cached = tensor.to(dtype)
            cast_cache[key] = cached
        return cached

    if not isinstance(lora_path, list):
        lora_path = [lora_path]
    
    if lora_multi is None:
        lora_multi = [1. for _ in lora_path]
    try:
        max_reserved_loras_mb = float(maxReservedLoras)
    except Exception:
        max_reserved_loras_mb = -1
    if max_reserved_loras_mb is None:
        max_reserved_loras_mb = -1
    pinned_total_mb = 0.0
    loras_nos = []
    loras_multi = []
    new_lora_path = []
    errors  = []
    adapters = {}
    adapter_no = 0
    pinned_sd_list = []
    pinned_names_list = []
    pinned_tied_weights_list = []
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
        tied_weights = None
        state_dict = safetensors2.torch_load_file(path, writable_tensors= False)

        if preprocess_sd != None:
            state_dict = preprocess_sd(state_dict)

        clean_up = False
        first_key = next(iter(state_dict), None)
        if first_key is  None:
            msg = f"Empty Lora '{path}'"
            error_msg = append(error_msg, msg) 
            fail = True
        else:
            prefixes = ("diffusion_model.", "transformer.")
            pos = first_key.find(".")
            prefix = first_key[0:pos+1]
            if prefix in prefixes:
                new_state_dict = {}
                for k, v in state_dict.items():
                    for candidate in prefixes:
                        if k.startswith(candidate):
                            k = k[len(candidate) :]
                            break
                    new_state_dict[k] = v
                state_dict = new_state_dict
                del new_state_dict

        if not fail:
            lokr_split_chunks = {}
            if split_linear_modules_map != None:
                new_state_dict = {}
                suffixes = [(".alpha", -2, False), (".lora_B.weight", -3, True), (".lora_A.weight", -3, False), (".lora_up.weight", -3, True), (".lora_down.weight", -3, False),(".dora_scale", -2, False),]
                for module_name, module_data in state_dict.items():
                    if module_name.endswith(".lokr_w1") or module_name.endswith(".lokr_w2"):
                        module_name = _split_lokr_key(module_name, module_data, state_dict, new_state_dict, lokr_split_chunks)
                        if module_name is not None: new_state_dict[module_name] = module_data
                    else:
                        name_parts = module_name.split(".")
                        for suffix, pos, any_split in suffixes:
                            if not module_name.endswith(suffix) or (map := split_linear_modules_map.get(name_parts[pos], None)) is None:
                                continue
                            parent_module_name = ".".join(name_parts[:pos])
                            sub_data = torch.split(module_data, map["split_sizes"], dim=0) if any_split else (module_data,) * len(map["mapped_modules"])
                            for sub_name, subdata in zip(map["mapped_modules"], sub_data):
                                new_state_dict[parent_module_name + "." + sub_name + suffix] = subdata
                            break
                        else:
                            new_state_dict[module_name] = module_data
                state_dict = new_state_dict
                del new_state_dict
            clean_up = True

            keys = list(state_dict.keys())
            dtype_cast_cache = {}

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
            dora_found = False
            lokr_found = False
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                lora_A = lora_B = diff_b = diff = lora_key = dora_scale = lokr_w1 = lokr_w2 = None
                adapter_type = "lora"
                if k.endswith(".diff"):
                    diff = v
                    module_name = k[ : -5]
                    adapter_type = "diff"
                elif k.endswith(".diff_b"):
                    diff_b = v
                    module_name = k[ : -7]
                    adapter_type = "lora"
                elif k.endswith(".dora_scale"):
                    dora_scale = v
                    module_name = k[ : -11]
                    adapter_type = "dora"
                elif k.endswith(".lokr_w1"):
                    lokr_w1 = v
                    module_name = k[ : -8]
                    adapter_type = "lokr"
                elif k.endswith(".lokr_w2"):
                    lokr_w2 = v
                    module_name = k[ : -8]
                    adapter_type = "lokr"
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
                    v = lora_A = _to_dtype_cached(lora_A, module.weight.dtype, dtype_cast_cache)
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
                    v = lora_B = _to_dtype_cached(lora_B, module.weight.dtype, dtype_cast_cache)
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
                    v = lora_B = _to_dtype_cached(lora_B, module.weight.dtype, dtype_cast_cache)
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
                    v = diff_b = _to_dtype_cached(diff_b, module.weight.dtype, dtype_cast_cache)
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
                    v = dora_scale = _to_dtype_cached(dora_scale, module.weight.dtype, dtype_cast_cache)
                elif lokr_w1 is not None:
                    out_dim, in_dim = int(module_shape[0]), int(module_shape[1])
                    m1, n1 = int(lokr_w1.shape[0]), int(lokr_w1.shape[1])
                    lokr_chunk = lokr_split_chunks.get(module_name, None) 
                    if lokr_chunk is None:
                        invalid = m1 <= 0 or n1 <= 0 or out_dim % m1 != 0 or in_dim % n1 != 0
                    else:
                        invalid = m1 <= 0 or n1 <= 0 or (lokr_chunk[1] - lokr_chunk[0]) != out_dim or in_dim % n1 != 0
                    if invalid:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': LoKr W1 dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lokr_w1 = {v.shape})."
                            error_msg = append(error_msg, msg)
                            fail = True
                        break
                    v = lokr_w1 = _to_dtype_cached(lokr_w1, module.weight.dtype, dtype_cast_cache)
                elif lokr_w2 is not None:
                    out_dim, in_dim = int(module_shape[0]), int(module_shape[1])
                    m2, n2 = int(lokr_w2.shape[0]), int(lokr_w2.shape[1])
                    lokr_chunk = lokr_split_chunks.get(module_name, None) 
                    if lokr_chunk is None:
                        invalid = m2 <= 0 or n2 <= 0 or out_dim % m2 != 0 or in_dim % n2 != 0
                    else:
                        invalid = m2 <= 0 or n2 <= 0 or (lokr_chunk[1] - lokr_chunk[0]) != out_dim or in_dim % n2 != 0
                    if invalid:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': LoKr W2 dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lokr_w2 = {v.shape})."
                            error_msg = append(error_msg, msg)
                            fail = True
                        break
                    v = lokr_w2 = _to_dtype_cached(lokr_w2, module.weight.dtype, dtype_cast_cache)
                if not check_only:
                    new_state_dict[k] = v
                    v = None
                    loras_module_data = loras_model_data.get(module, None)
                    assert loras_module_data is not None
                    loras_adapter_data, meta = _ensure_adapter_meta(loras_module_data, adapter_name)
                    if adapter_type in ("dora", "lokr", "diff"):
                        meta["type"] = adapter_type
                    elif "type" not in meta:
                        meta["type"] = "lora"
                    if adapter_type == "lokr":
                        lokr_found = True
                        if lokr_chunk is not None: meta["lokr_chunk"] = lokr_chunk
                    elif adapter_type == "dora":
                        dora_found = True
                    if lora_A != None:
                        loras_adapter_data[0] = lora_A
                    elif lora_B != None:
                        loras_adapter_data[1] = lora_B 
                    elif dora_scale != None:
                        loras_adapter_data[3] = dora_scale 
                    elif lokr_w1 is not None:
                        loras_adapter_data[0] = lokr_w1
                    elif lokr_w2 is not None:
                        loras_adapter_data[1] = lokr_w2
                    else:
                        loras_adapter_data[2] = diff_b 
                    if rank != None and lora_key is not None and "lora" in lora_key:
                        alpha_key = k[:-len(lora_key)] + "alpha"
                        alpha = lora_alphas.get(alpha_key, None)
                        if alpha is not None: loras_adapter_data[4] = alpha / rank 
            lora_A = lora_B = diff = diff_b = v = loras_module_data = loras_adapter_data = lora_alphas = dora_scale = lokr_w1 = lokr_w2 = dtype_cast_cache = None

            if len(invalid_keys)  > 0:
                msg = f"Lora '{path}' contains non Lora keys '{trunc(invalid_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
            if len(unexpected_keys)  > 0:
                msg = f"Lora '{path}' contains unexpected module keys, it is likely that this Lora is for a different model : '{trunc(unexpected_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
            if (not fail) and (not check_only) and dora_found and lokr_found:
                msg = f"Lora '{path}': LoKr cannot be mixed with DoRA on the same module."
                error_msg = append(error_msg, msg)
                fail = True
            if (not fail) and lokr_found:
                for module, loras_module_data in loras_model_data.items():
                    if adapter_name not in loras_module_data: continue
                    loras_adapter_data, meta = _ensure_adapter_meta(loras_module_data, adapter_name)
                    adapter_type = meta.get("type", "lora")
                    if adapter_type != "lokr": continue
                    w1, w2 = loras_adapter_data[0], loras_adapter_data[1]
                    if w1 is not None and w2 is not None:
                        lokr_chunk = meta.get("lokr_chunk", None)
                        effective_chunk = lokr_chunk if lokr_chunk is not None else (0, int(w1.shape[0]) * int(w2.shape[0]))
                        meta["lokr_specs"] = _build_lokr_chunk_specs(effective_chunk, int(w2.shape[0]))
                        module_shape = module.weight.shape
                        if lokr_chunk is None:
                            invalid = int(module_shape[0]) != int(w1.shape[0]) * int(w2.shape[0]) or int(module_shape[1]) != int(w1.shape[1]) * int(w2.shape[1])
                        else:
                            invalid = int(module_shape[0]) != (lokr_chunk[1] - lokr_chunk[0]) or int(module_shape[1]) != int(w1.shape[1]) * int(w2.shape[1])
                        if invalid:
                            module_name = module_names.get(module, type(module).__name__)
                            msg = f"Lora '{path}/{module_name}': LoKr factors are not compatible with model shape (model = {module_shape}, lokr_w1 = {tuple(w1.shape)}, lokr_w2 = {tuple(w2.shape)})."
                            error_msg = append(error_msg, msg)
                            fail = True
                            break
            if (not fail) and (not check_only):
                alias_owners = {}
                for module, loras_module_data in loras_model_data.items():
                    loras_adapter_data = loras_module_data.get(adapter_name, None)
                    if loras_adapter_data is None:
                        continue
                    module_name = module_names.get(module, None)
                    if module_name is None:
                        continue
                    for slot in range(min(4, len(loras_adapter_data))):
                        tensor = loras_adapter_data[slot]
                        if not torch.is_tensor(tensor):
                            continue
                        key = (slot, _get_tensor_ref(tensor))
                        owner_name = alias_owners.get(key, None)
                        if owner_name is None:
                            alias_owners[key] = module_name
                        else:
                            loras_adapter_data[slot] = owner_name + "#" + str(slot)
            if (not fail) and (not check_only) and pinnedLora:
                tied_weights = _extract_tie_weights_from_sd(new_state_dict, path, verboseLevel=max(0, verboseLevel))
        if fail or skip:
            if fail:
                errors.append((path, error_msg))
                print(error_msg)
            if clean_up and not check_only:
                for _, loras_module_data in loras_model_data.items():
                    if adapter_name in loras_module_data:
                        del loras_module_data[adapter_name]

        else:
            if not check_only:
                # model._loras_tied_weights[adapter_name] = tied_weights
                if pinnedLora:
                    if max_reserved_loras_mb < 0:
                        pinned_sd_list.append(new_state_dict)
                        pinned_names_list.append(path)
                        pinned_tied_weights_list.append(tied_weights)
                    else:
                        lora_size_mb = _state_dict_size_mb(new_state_dict)
                        if pinned_total_mb + lora_size_mb <= max_reserved_loras_mb:
                            pinned_sd_list.append(new_state_dict)
                            pinned_names_list.append(path)
                            pinned_tied_weights_list.append(tied_weights)
                            pinned_total_mb += lora_size_mb

            del state_dict 


            adapters[adapter_name] = path
            loras_nos.append(adapter_name)
            new_lora_path.append(path)        
            loras_multi.append(1.0 if i > (len(lora_multi) -1) else lora_multi[i])
            adapter_no += 1
            if verboseLevel >=1:
                if check_only:
                    print(f"Lora '{path}' was found for model '{_get_module_name(model)}'")
                else:
                    print(f"Lora '{path}' was loaded in model '{_get_module_name(model)}'")
    
    model._loras_errors = errors
    if not check_only:
        if pinnedLora and len(pinned_sd_list) > 0:
            _pin_sd_to_memory(pinned_sd_list, pinned_names_list, pinned_tied_weights_list)
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
    target = getattr(model, "_lora_owner", None)
    while target is not None and target is not model:
        model = target
        target = getattr(model, "_lora_owner", None)
    model._lora_step_no = step_no
    sub_loras = getattr(model, "_subloras", None)
    if sub_loras:
        submodules = sub_loras.values() if isinstance(sub_loras, dict) else sub_loras
        for submodule in submodules:
            if submodule is model:
                continue
            submodule._lora_step_no = step_no

def activate_loras(model, lora_nos, lora_multi = None):
    target = getattr(model, "_lora_owner", None)
    while target is not None and target is not model:
        model = target
        target = getattr(model, "_lora_owner", None)

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
    sub_loras = getattr(model, "_subloras", None)
    if sub_loras:
        submodules = sub_loras.values() if isinstance(sub_loras, dict) else sub_loras
        for submodule in submodules:
            if submodule is model:
                continue
            submodule._lora_step_no = 0
            submodule._loras_active_adapters = lora_nos
            submodule._loras_scaling = lora_scaling_dict


def move_loras_to_device(model, device="cpu" ):
    if hasattr( model, "_lora_loadable_modules"):
        for k in model._lora_loadable_modules:
            move_loras_to_device(getattr(model,k), device)
        return
    
    for k, m in model.named_modules():
        if ".lora_" in k:
            m.to(device)

def fast_load_transformers_model(model_path: str,  do_quantize = False, quantizationType =  qint8, pinToMemory = False, partialPinning = False, forcedConfigPath = None, defaultConfigPath = None, modelClass=None, modelPrefix = None, writable_tensors = True, verboseLevel = -1, preprocess_sd  = None, fused_split_map = None, modules = None,  return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, configKwargs ={}):
    """
    quick version of .LoadfromPretrained of  the transformers library
    used to build a model and load the corresponding weights (quantized or not)
    """       

    
    import os.path
    if not isinstance(model_path, list):
        model_path = [model_path]


    if not builtins.all(file_name.endswith(".sft") or file_name.endswith(".safetensors") or file_name.endswith(".pt") or file_name.endswith(".bin") or file_name.endswith(".ckpt") or get_extension_handler(file_name) is not None for file_name in model_path):
        raise Exception(f"File Extension of file {model_path} is not supported")

    model_path = [ _get_model(file) for file in model_path] 
    if any( file == None for file in model_path):
        raise Exception(f"Unable to find file {model_path}")
    
    verboseLevel = _compute_verbose_level(verboseLevel)
    if model_path[-1].endswith(".sft") or model_path[-1].endswith(".safetensors"):
        with safetensors2.safe_open(model_path[-1], writable_tensors =writable_tensors) as f:
            metadata = f.metadata() 
    else:
        metadata = None

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

    load_model_data(model,model_path, do_quantize = do_quantize, quantizationType = quantizationType, pinToMemory= pinToMemory, partialPinning= partialPinning, modelPrefix = modelPrefix, writable_tensors =writable_tensors, preprocess_sd = preprocess_sd, fused_split_map = fused_split_map, modules = modules, return_shared_modules =  return_shared_modules, default_dtype = default_dtype, ignore_unused_weights = ignore_unused_weights, verboseLevel=verboseLevel )

    return model

def flush_torch_caches():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except torch.cuda.CudaError:
            pass
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass
    if os.name == "nt":
        try:
            import ctypes, ctypes.wintypes as wintypes, os as _os
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_QUERY_INFORMATION = 0x0400
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, _os.getpid())
            if handle:
                psapi.EmptyWorkingSet(handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass
    from accelerate import init_empty_weights
    with init_empty_weights():
        for _ in range(3):
            dummy_tensor = torch.nn.Embedding(256384, 1024)
            dummy_tensor = None    


def map_state_dict(state_dict, rules):

    def map_one_sd(sd):
        if sd is None: return None
        for rule, repl in rules.items():
            new_sd= {}
            new_start = len(rule)
            prefix = rule + "."
            for k,v in sd.items():
                if k.startswith(prefix):
                    if repl is not None:
                        if len(repl) == 0:
                            k= k[new_start+1:]
                        else:            
                            k = repl + k[new_start:]
                        if isinstance(v, list):
                            new_v = []
                            for sub in v:
                                if sub.startswith(prefix):
                                    if len(repl) == 0:
                                        sub= sub[new_start+1:]
                                    else:            
                                        sub = repl + sub[new_start:]
                                new_v.append(sub)
                            v = new_v

                        new_sd[k] = v
                else:
                    new_sd[k] = v
            sd = new_sd
        return sd
    
    if isinstance(state_dict, list):
        return [map_one_sd(sd) for sd in state_dict]
    else:
        return map_one_sd(state_dict)

def filter_state_dict_basic(state_dict, base_model_prefix, keep_prefix = False):
    new_state_dict= {}
    start = -1
    if keep_prefix:
        for k,v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_state_dict[k] = v
    else:
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

def load_sd(file_path, filters = None, keep_prefixes = False, writable_tensors = True):
    state_dict, metadata = _safetensors_load_file(file_path, writable_tensors =writable_tensors)
    quantization_map = None
    tied_weights_map = None
    if metadata !=  None:
        quantization_map = metadata.get("quantization_map", None)
        tied_weights_map = metadata.get("tied_weights_map", None)

    if filters is not None:
        if not isinstance(filters, list): filters = [filters]
        new_sd = {}
        new_quantization_map = {}
        new_tied_weights_map = {}
        for one_filter in filters:
            new_sd.update(filter_state_dict_basic(state_dict, one_filter, keep_prefixes))
            if quantization_map is not None:
                new_quantization_map.update(filter_state_dict_basic(quantization_map, one_filter, keep_prefixes))
            if tied_weights_map is not None:
                new_tied_weights_map.update(filter_state_dict_basic(tied_weights_map, one_filter, keep_prefixes))
        state_dict = new_sd
        quantization_map = new_quantization_map if len(new_quantization_map) else None
        tied_weights_map = new_tied_weights_map if len(new_tied_weights_map) else None
    return state_dict, quantization_map, tied_weights_map


@cudacontext("cpu")
def load_model_data(model, file_path, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, modelPrefix = None, writable_tensors = True,  preprocess_sd = None, postprocess_sd = None, fused_split_map = None, modules = None, return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, verboseLevel = -1, ignore_missing_keys = False):
    """
    Load a model, detect if it has been previously quantized using quanto and do the extra setup if necessary
    """
    if isinstance(preprocess_sd, dict):
        preprocess_fn = lambda sd, qm, twm: map_state_dict([sd, qm, twm], rules=preprocess_sd)
    else:
        preprocess_fn = preprocess_sd 
    if not isinstance(file_path, list):
        file_path = [file_path]

    file_count =  len(file_path)
    if isinstance(modules, (list,str)):
        if isinstance(modules, str): modules = [modules]
        file_path += modules
        modules = None

    normalized_paths = []
    for file in file_path:
        if isinstance(file, (dict, tuple)):
            normalized_paths.append(file)
        else:
            resolved = _get_model(file)
            if isinstance(resolved, str):
                resolved = normalize_extension_path(resolved)
            normalized_paths.append(resolved)
    file_path = normalized_paths
    if any(file is None for file in file_path):
        raise Exception(f"Unable to find file {file_path}")
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
        hybrid_quantization_map = False
        tied_weights_map = None
        metadata = None
        detected_kind = None
        if isinstance(file, tuple):
            if len(file)==2:
                state_dict, quantization_map = file
            elif len(file)==3:
                state_dict, quantization_map, tied_weights_map = file
            else:
                raise Exception("Expected a tuple of (state_dict, quantization_map, tied_weights_map)")
        elif isinstance(file, dict):
            state_dict = file
        elif isinstance(file, str) and get_extension_handler(file) is not None:
            ext_handler = get_extension_handler(file)
            load_fn = getattr(ext_handler, "load_state_dict", None)
            if not callable(load_fn):
                ext = os.path.splitext(file)[1].lower().lstrip(".")
                raise Exception(f"Missing load_state_dict for *.{ext} handler")
            result = load_fn( file, writable_tensors=writable_tensors, verboseLevel=verboseLevel, default_dtype=default_dtype, pin_to_memory=pinToMemory, )
            if isinstance(result, tuple):
                if len(result) == 2:
                    state_dict, quantization_map = result
                elif len(result) == 3:
                    state_dict, quantization_map, tied_weights_map = result
                else:
                    raise Exception("Expected a tuple of (state_dict, quantization_map, tied_weights_map)")
            else:
                state_dict = result
        elif not (".safetensors" in file or ".sft" in file):
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

        if metadata !=  None:
            if quantization_map is None:
                quantization_map = metadata.get("quantization_map", None)
            config = metadata.get("config", None)
            if config is not None:
                model._config = config
            if tied_weights_map is None:
                tied_weights_map = metadata.get("tied_weights_map", None)

        if quantization_map is None and isinstance(file, str):
            pos = str.rfind(file, ".")
            if pos > 0:
                quantization_map_path = file[:pos]
            quantization_map_path += "_map.json"

            if os.path.isfile(quantization_map_path):
                with open(quantization_map_path, 'r') as f:
                    quantization_map = json.load(f)

        if preprocess_fn != None:
            num_params = len(inspect.signature(preprocess_fn).parameters)
            state_dict = preprocess_fn(*[state_dict, quantization_map, tied_weights_map][:num_params])
            if isinstance(state_dict, (tuple,list)):
                if len(state_dict)==2: 
                    state_dict, quantization_map = state_dict
                else:
                    state_dict, quantization_map, tied_weights_map = state_dict
            hybrid_quantization_map = quantization_map is not None

        if tied_weights_map != None:
            for name, tied_weights_list in tied_weights_map.items():
                mapped_weight = state_dict[name]
                for tied_weights in tied_weights_list:
                    state_dict[tied_weights] = mapped_weight


        if quantization_map is None or hybrid_quantization_map :
            conv_result = detect_and_convert(state_dict, default_dtype=default_dtype, verboseLevel=verboseLevel)
            detected_kind = conv_result.get("kind")
            if conv_result.get("kind") not in ("none", "quanto"):
                state_dict = conv_result["state_dict"]
                quantization_map = quantization_map or {}
                quantization_map.update(conv_result["quant_map"])
                conv_result = None
                # enable_fp8_fp32_scale_support()

            if detected_kind in (None, "none") and isinstance(file, str) and (".safetensors" in file or ".sft" in file):
                try:
                    info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel)
                    detected_kind = info.get("kind")
                except Exception:
                    detected_kind = detected_kind or None
            if detected_kind not in (None, "none") and isinstance(file, str):
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
        num_params = len(inspect.signature(postprocess_sd).parameters)
        state_dict = postprocess_sd(*[state_dict, quantization_map, tied_weights_map][:num_params])
        if isinstance(state_dict, (tuple,list)):
            if len(state_dict)==2: 
                state_dict, quantization_map = state_dict
            else:
                state_dict, quantization_map, tied_weights_map = state_dict
        
    if modelPrefix != None:
        base_model_prefix = modelPrefix + "."
        state_dict = filter_state_dict_basic(state_dict,base_model_prefix)
        if quantization_map != None:
            quantization_map = filter_state_dict_basic(quantization_map,base_model_prefix)
        if tied_weights_map != None:
            tied_weights_map = filter_state_dict_basic(tied_weights_map,base_model_prefix)

    if fused_split_map:
        state_dict, quantization_map = split_fused_weights(
            state_dict,
            quantization_map,
            fused_split_map,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
        )

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
        if any(isinstance(file, str) and "quanto" in file for file in file_path) and not do_quantize:
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
            if not ignore_missing_keys:
                raise Exception(f"Missing keys: {missing_keys}")
        else:
            state_dict = filter_state_dict_basic(state_dict, base_model_prefix)
            missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
            if len(missing_keys) > 0 and not ignore_missing_keys:
                raise Exception(f"Missing keys: {missing_keys}")
        
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
    
def _mm_lora_linear_forward(module, *args, **kwargs):
    if args:
        inp = args[0]
    else:
        inp = kwargs.get("input", None)
    weight = getattr(module, "weight", None)
    if torch.is_tensor(inp) and torch.is_tensor(weight):
        if inp.dtype != weight.dtype and inp.dtype.is_floating_point and weight.dtype.is_floating_point:
            inp = inp.to(weight.dtype)
            if args:
                args = (inp,) + args[1:]
            else:
                kwargs = dict(kwargs)
                kwargs["input"] = inp
    loras_data = getattr(module, "_mm_lora_data", None)
    if not loras_data:
        return module._mm_lora_old_forward(*args, **kwargs)
    if not hasattr(module, "_mm_manager"):
        pass
    return module._mm_manager._lora_linear_forward(
        module._mm_lora_model,
        module,
        loras_data,
        *args,
        **kwargs,
    )


def _mm_lora_generic_forward(module, *args, **kwargs):
    loras_data = getattr(module, "_mm_lora_data", None)
    if not loras_data:
        return module._mm_lora_old_forward(*args, **kwargs)
    return module._mm_manager._lora_generic_forward(
        module._mm_lora_model,
        module,
        loras_data,
        module._mm_lora_old_forward,
        *args,
        **kwargs,
    )


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


            if tied_param is None:
                blocks_params_size +=  param_size
                self.parameters_ref[ref] = (submodule, k)

        for k, p in submodule.named_buffers(recurse=False):
            ref = _get_tensor_ref(p)
            tied_param =  self.parameters_ref.get(ref, None)
            blocks_params.append( (submodule, k, p, True, tied_param) )
            if tied_param is None:
                blocks_params_size += p.data.nbytes
                self.parameters_ref[ref] = (submodule, k)

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

    def _move_loras(self, loras_active_adapters, loras_modules, to_GPU, model=None):
        shortcuts = getattr(model, "_loras_model_shortcuts", None) if model is not None else None
        tensor_cache = {}
        alias_cache = {}

        def _move_tensor(item):
            if item is None:
                return None
            if torch.is_tensor(item):
                ref = _get_tensor_ref(item)
                moved = tensor_cache.get(ref, None)
                if moved is None:
                    moved = item.cuda(non_blocking=True)
                    tensor_cache[ref] = moved
                return moved
            return item

        def _resolve_alias(value, adapter):
            if not isinstance(value, str):
                return _move_tensor(value)
            cache_key = (adapter, value)
            cached = alias_cache.get(cache_key, None)
            if cached is not None:
                return cached
            while isinstance(value, str):
                target, slot = value.rsplit("#", 1)
                value = shortcuts[target][adapter][int(slot)]
            resolved = _move_tensor(value)
            alias_cache[cache_key] = resolved
            return resolved

        for _, lora_module in loras_modules.items():
            for adapter in loras_active_adapters:
                lora_data = lora_module.get(adapter, None)
                if lora_data is None:
                    continue
                key = adapter + '_GPU'
                if to_GPU:
                    moved_data = list(lora_data)
                    for i in range(min(4, len(moved_data))):
                        moved_data[i] = _resolve_alias(moved_data[i], adapter)
                    lora_module[key] = moved_data
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
                        q = _make_buffer(q)
                    else:
                        q = _make_parameter(q, requires_grad=False)                        
                    setattr(parent_module, n , q)

                    if tied_param != None:
                        setattr( tied_param[0], tied_param[1], q) 
                    del p, q
                    if loras_model_data != None:
                        lora_data =  loras_model_data.get(parent_module, None)
                        if lora_data != None:
                            loras_modules[parent_module]= lora_data
                if len(loras_modules) > 0:
                    self._move_loras(loras_active_adapters, loras_modules, True, model)

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
                q = _make_buffer(p)
            else:
                q = _make_parameter(p, requires_grad=False)
            setattr(parent_module, n , q)
            del p, q 

            if loras_model_data != None:
                lora_data =  loras_model_data.get(parent_module, None)
                if lora_data != None:
                    loras_modules[parent_module]= lora_data

        if len(loras_modules) > 0:
            self._move_loras(loras_active_adapters, loras_modules, False, model)

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
            diff_w, _, diff_b, _, alpha = data[:5]
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
        adapters_data,                # dict: name+"_GPU" -> [A, B, diff_b, g_abs, alpha, meta?]; g_abs=None means LoRA
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
            A, B, diff_b, g_abs, alpha = data[:5]
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

    def _lokr_chunk_forward(self, x_2d, lokr_w1, lokr_w2, specs=None):
        n1 = int(lokr_w1.shape[1])
        n2 = int(lokr_w2.shape[1])
        if len(specs) == 0:
            return x_2d.new_zeros((x_2d.shape[0], 0))

        x_3d = x_2d.view(-1, n1, n2)
        pieces = []
        for r0, r1, c0, c1 in specs:
            y = torch.matmul(lokr_w1[r0:r1].unsqueeze(0), x_3d)
            pieces.append(torch.matmul(y, lokr_w2[c0:c1].T).reshape(-1, (r1 - r0) * (c1 - c0)))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)



    def _lora_linear_forward(self, model, submodule, loras_data, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = submodule.weight
        bias = submodule.bias
        active_adapters = model._loras_active_adapters
        loras_scaling = model._loras_scaling
        any_dora = False
        any_lokr = False
        for active_adapter in active_adapters:
            data = loras_data.get(active_adapter + '_GPU', None)
            if data is None:
                continue
            if data[3] is not None:
                any_dora = True
            meta = data[5]
            if meta.get("type", "lora") == "lokr":
                any_lokr = True
            if any_dora and any_lokr:
                break

        dtype = weight.dtype
        if any_dora and not any_lokr: # sum base weight and lora matrices instead of applying input on each sub lora matrice if input is too large. This will save a lot VRAM and compute
            original_bias = True
            if isinstance(submodule, QModuleMixin):
                weight = weight.view(weight.shape) # get a persistent copy of the on the fly dequantized weights
            else:
                weight = weight.clone()
            for active_adapter in active_adapters:
                data = loras_data.get(active_adapter + '_GPU', None)
                if data is None:
                    continue
                lora_A_weight, lora_B_weight, diff_b, g_abs, alpha = data[:5]
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
                if lora_A_weight is not None:
                    weight.addmm_(lora_B_weight, lora_A_weight, alpha=scaling)
                if diff_b is not None:
                    if bias is None:
                        bias = diff_b.clone()
                        original_bias = False
                    elif original_bias:
                        bias = bias.clone()
                        original_bias = False
                    bias.add_(diff_b, alpha=scaling)
            weight, bias = self._dora_linear_forward(model, submodule, loras_data, weight, bias, original_bias)
            base_bias = bias
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            result = torch.nn.functional.linear(x, weight, bias=base_bias)

        else:
            base_bias = bias
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            result = torch.nn.functional.linear(x, weight, bias=base_bias)

            if active_adapters:
                compute_dtype = result.dtype
                if result.dtype != compute_dtype:
                    result = result.to(compute_dtype)
                x = x.to(compute_dtype)
                x_2d = x.reshape(-1, x.shape[-1])
                result_2d = result.reshape(-1, result.shape[-1])

                for active_adapter in active_adapters:
                    data = loras_data.get(active_adapter + '_GPU', None)
                    if data is None:
                        continue
                    lora_A, lora_B, diff_b, g_abs, alpha, adapter_meta = data
                    adapter_type = adapter_meta.get("type", "lora")
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

                    if adapter_type == "lokr":
                        if lora_A is None or lora_B is None:
                            continue
                        n1 = int(lora_A.shape[1])
                        n2 = int(lora_B.shape[1])
                        out_dim = int(lora_A.shape[0]) * int(lora_B.shape[0])
                        chunk = adapter_meta.get("lokr_chunk", None)
                        if chunk is None:
                            x_3d = x_2d.view(-1, n1, n2)
                            y = torch.matmul(lora_A.unsqueeze(0), x_3d)
                            y = torch.matmul(y, lora_B.T)
                            result_2d.add_(y.reshape(-1, out_dim), alpha=scaling)
                        else:
                            specs = adapter_meta.get("lokr_specs", [])
                            y = self._lokr_chunk_forward(x_2d, lora_A, lora_B, specs=specs)
                            result_2d.add_(y, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        del y
                        continue
                    if adapter_type == "diff":
                        result_2d.addmm_(x_2d, lora_B.T, beta=1, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        continue

                    if lora_A is None:
                        result_2d.add_(diff_b, alpha=scaling)
                    else:
                        y = x_2d @ lora_A.T
                        result_2d.addmm_(y, lora_B.T, beta=1, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        del y
                target_dtype = dtype
                if result.dtype != target_dtype:
                    result = result.to(target_dtype)

        return result


    def hook_lora(self, submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name):
        old_forward = submodule.forward

        loras_data = {}
        assert submodule_name not in loras_model_shortcuts 
        loras_model_shortcuts[submodule_name] = loras_data
        loras_model_data[submodule] = loras_data
        submodule._mm_lora_data = loras_data
        submodule._mm_lora_model = current_model
        submodule._mm_lora_old_forward = old_forward

        if isinstance(submodule,  torch.nn.Linear) or getattr(submodule, "is_nvfp4", False):
            target_fn = _mm_lora_linear_forward
        else:
            target_fn = _mm_lora_generic_forward
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
        weight = getattr(target_module, "weight", None)
        weight_qtype = getattr(weight, "qtype", None) if weight is not None else None
        qint4quantization = isinstance(target_module, QModuleMixin) and weight_qtype == qint4
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
        setattr(target_module, "_mm_manager", self)
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
                    del sub_tensors
                else:
                    if not ignore_dtype:
                        dtype = p.data.dtype
                        if convertWeightsFloatTo != None and dtype == torch.float32 :
                            # convert any left overs float32 weight to bfloat16 / float16 to divide by 2 the model memory footprint
                            dtype = convertWeightsFloatTo if model_dtype == None else model_dtype
                            if dtype != torch.float32:
                                p.data = p.data.to(dtype)
                        if model_dtype is None:
                            model_dtype = dtype
                        else:
                            if model_dtype != dtype:
                                pass
                            assert model_dtype == dtype
                    current_model_size +=  torch.numel(p.data) * p.data.element_size()
        if model_dtype is None:
            model_dtype = convertWeightsFloatTo if convertWeightsFloatTo is not None else torch.bfloat16
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
        current_model._force_device= "cuda"
        towers_names, towers_modules = _detect_main_towers(current_model)
        compilationInThisOne = compileAllModels or model_id in modelsToCompile 
                
        if pinAllModels or model_id in modelsToPin:
            if hasattr(current_model,"_already_pinned"):
                if self.verboseLevel >=1:
                    print(f"Model '{model_id}' already pinned to reserved memory")
            else:
                _pin_to_memory(current_model, model_id, partialPinning= partialPinning, pinnedPEFTLora = pinnedPEFTLora, perc_reserved_mem_max = perc_reserved_mem_max, verboseLevel=verboseLevel)            

        current_budget = getattr(current_model, "_budget", model_budgets[model_id])
        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False
        self.loaded_blocks[model_id] = None
        any_lora =  loras !=None and model_id in loras
        if any_lora: 
            loras_model_data, loras_model_shortcuts = {}, {}
            current_model._loras_model_data = loras_model_data 
            current_model._loras_model_shortcuts = loras_model_shortcuts
        modules_to_be_compiled, modules_names_to_be_compiled = [], []
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
            compile_me = getattr(submodule, "_compile_me", None)
            for submodule_method_name in submodule_method_names:
                if not hasattr(submodule, submodule_method_name ): continue
                if submodule_method_name == "forward" and any_lora and hasattr(submodule,"weight"):
                    submodule_method = self.hook_lora(submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name)                
                else:
                    submodule_method = getattr(submodule, submodule_method_name)
                if callable(submodule_method):
                    if top_submodule and cur_blocks_name is None and any_lora and len(submodule._parameters):
                        pass
                    if top_submodule and cur_blocks_name is None and not (any_lora and len(submodule._parameters)):
                        self.hook_change_module(submodule, current_model, model_id, submodule_name, submodule_method, submodule_method_name)
                    elif compilationInThisOne and submodule in towers_modules and not compile_me == False or compile_me == True: 
                        self.hook_preload_blocks_for_compilation(submodule, model_id, cur_blocks_name, context = submodule_name )
                        compile_item_name= ".".join(submodule_name.split(".")[:-1]) if submodule_name[-1].isdigit() else submodule_name
                        if compile_item_name not in modules_names_to_be_compiled: modules_names_to_be_compiled.append(compile_item_name)
                        modules_to_be_compiled.append(submodule)
                    else:
                        if compilationInThisOne: #and False
                            self.hook_check_load_into_GPU_if_needed(submodule, current_model, model_id, cur_blocks_name, submodule_method, context = submodule_name )
                        else:
                            self.hook_check_load_into_GPU_if_needed_default(submodule, current_model, model_id, cur_blocks_name, submodule_method, context = submodule_name )

                    self.add_module_to_blocks(model_id, cur_blocks_name, submodule, prev_blocks_name, submodule_name)


        # compile main iterative modules stacks ("towers")
        if compilationInThisOne:
            if self.verboseLevel>=1:
                if len(modules_names_to_be_compiled)>0:
                    formated_compiled_names = [name + '.*' for name in modules_names_to_be_compiled]
                    print(f"Pytorch compilation of '{model_id}' is scheduled for these modules : {formated_compiled_names}.")
                else:
                    print(f"Pytorch compilation of model '{model_id}' is not yet supported.")

            for submodel in modules_to_be_compiled:
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
