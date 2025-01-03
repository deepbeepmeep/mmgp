
<p align="center">
  <H2>Memory Management 3.0 for the GPU Poor by DeepBeepMeep</H2>	
</p>


This module contains multiples optimisations so that models such as Flux (and derived), Mochi, CogView, HunyuanVideo, ...  can run smoothly on a 12 to 24 GB GPU limited card. 
This a replacement for the accelerate library that should in theory manage offloading, but doesn't work properly with models that are loaded / unloaded several
times in a pipe (eg VAE).

Requirements:
- VRAM: minimum 12 GB, recommended 24 GB (RTX 3090/ RTX 4090) 
- RAM: minimum 24 GB, recommended 48 GB 

This module features 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM).\
These RAM requirements are for Linux systems. Due to different memory management Windows will require an extra 16 GB of RAM to run the corresponding profile.

Each profile may use the following: 
- Low RAM consumption (thanks to a rewritten safetensors library) that allows low RAM on the fly quantization
- Smart automated loading / unloading of models in the GPU to avoid unloading models that may be needed again soon
- Smart slicing of models to reduce memory occupied by models in the VRAM
- Ability to pin models to reserved RAM to accelerate transfers to VRAM
- Async transfers to VRAM to avoid a pause when loading a new slice of a model
- Automated on the fly quantization or ability to load pre quantized models

## Installation
First you need to install the module in your current project with:
```shell
pip install mmgp
```


## Usage 

It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
1) First make sure that the pipeline explictly loads the models in the CPU device, for instance:
```
  pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
```

2) Once every potential Lora has been loaded and merged, add the following lines for a quick setup:
```
  from mmgp import offload, profile_type
  offload.profile(pipe, profile_type.HighRAM_LowVRAM_Fast)
```

You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM_Fastest: at least 48 GB of RAM and 24 GB of VRAM : the fastest well suited for a RTX 3090 / RTX 4090
- HighRAM_LowVRAM_Fast (recommended): at least 48 GB of RAM and 12 GB of VRAM : a bit slower, better suited for RTX 3070/3080/4070/4080 
            or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM_Medium: at least 32 GB of RAM and 24 GB of VRAM : so so speed but adapted for RTX 3090 / RTX 4090 with limited RAM
- LowRAM_LowVRAM_Slow: at least 32 GB of RAM and 12 GB of VRAM :  if have little VRAM or generate longer videos 
- VerylowRAM_LowVRAM_Slowest: at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work

By default the 'transformer' will be quantized to 8 bits for all profiles. If you don't want that you may specify the optional parameter *quantizeTransformer = False*.

Every parameter set automatically by a profile can be overridden with one or multiple parameters accepted by *offload.all* (see below):
```
  from mmgp import offload, profile_type
  offload.profile(pipe, profile_type.HighRAM_LowVRAM_Fast, budgets = 1000)
```
If you want to know which parameter are set by one specific profile you can use the parameter *verboseLevel=2*

## Alternatively you may want to create your own profile with specific parameters:

For example:
```
  from mmgp import offload
  offload.all(pipe, pinnedMemory=True, ExtraModelsToQuantize = ["text_encoder_2"] )
```  
- pinnedMemory: Boolean (for all models) or List of models ids to pin to RAM. Every model pinned to RAM will load much faster (up to 2 times) but this requires more RAM
- quantizeTransformer: boolean by default True. The 'transformer' model in the pipe contains usually the video or image generator is by defaut; quantized on the fly by default to 8 bits. If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. If you don't want to quantize the image generator, you need to set the option *quantizeTransformer* to *False* to turn off on the fly quantization.
- extraModelsToQuantize: list of additional modelids of models to quantize on the fly. If the corresponding model is already quantized, this option will be ignored.
- budgets: either a number in mega bytes (for all models, if 0 unlimited budget) or a dictionary that maps model ids to mega bytes : define the budget in VRAM (in fact the real number is 1.5 this number or 2.5 if asyncTransfers are also enabled) that is allocated in VRAM for each model. 
The smaller this number, the more VRAM left for image data / longer video but also the slower because there will be lots of loading / unloading between the RAM and the VRAM. If model is too big to fit in a budget, it will be broken down in multiples parts that will be unloaded / loaded consequently. The speed of low budget can be  increased (up to 2 times) by turning on the options pinnedMemory and asyncTransfers.
- asyncTransfers: boolean, load to the GPU the next model part while the current part is being processed. This requires twice the budget if any is defined. This may increase speed by 20% (mostly visible on fast modern GPUs).
- verboseLevel: number between 0 and 2 (1 by default), provides various level of feedback of the different processes
- perc_reserved_mem_max: a float below 0.5 (or 0 for auto), may be reduced to a lower number if any out of memory is triggered while using pinnedMemory
- compile (experimental): list of model ids to compile, may accelerate processing (or not) depending on the type of GPU. As of 01/01/2025 it will work only on Linux since compilation relies on Triton which is not yet supported on Windows

If you are short on RAM and plan to work with quantized models, it is recommended to load pre-quantized models direclty rather than using on the fly quantization (especially on Windows) as due to the way safetensors work almost twice the amount of RAM  may be needed for the loading of the model.

##  Going further

The module includes several tools to package a light version of your favorite video / image generator:
- *save_model(model, file_path, do_quantize = False, quantization_type = qint8 )*\
Save tensors of a model already loaded in memory in a safetensor format (much faster to reload). You can save it in a quantized format (default qint8 quantization recommended).
The resulting safetensor file will contain extra fields in its metadata such as the quantization map and its configuration, so you will be able to move the file around without files such as *config.json* or *file_map.json*.
You will need *load_model_data* or *fast_load_transformers_model* to read the file again . You may also load it using the default *safetensor* librar however you will need to provide in the same directory any complementary file that are usually requested (for instance *config.json*)

- *load_model_data(model, file_path: str, do_quantize = False, quantization_type = qint8, pinToRAM = False, partialPin = False)*\
Load the tensors data of a model in RAM of a model already initialized with no data. Detect and handle quantized models saved previously with *save_model*.A model can also be quantized on the fly while being loaded. The model which is loaded can be pinned to RAM while it is loaded, this is more RAM efficient than pinning tensors later using *offline.all* or *offline.profile*

- *fast_load_transformers_model(model_path: str, do_quantize = False, quantization_type = qint8, pinToRAM = False, partialPin = False)*\
Initialize (build the model hierarchy in memory) and fast load the corresponding tensors of a 'transformers' or 'diffusers' library model.
The advantages over the original *from_pretrained* method is that a full model can fit into a single file with a filename of your choosing (thefore you can have multiple 'transformers' versions of the same model in the same directory) and prequantized models are processed in a transparent way. 
Last but not least, you can also on the fly pin to RAM the whole model or the most important part of it (partialPin = True) in a more efficient way (faster and requires less RAM) than if you did through *offload.all* or *offload.profile*.



The typical workflow wil be:
1) temporarly insert the *save_model* function just after a model has been fully loaded to save a copy of the model / quantized model.
2) replace the full initalizing / loading logic with *fast_load_transformers_model* (if there is a *from_pretrained* call to a transformers object) or only the tensor loading functions (*torch.load_model_file* and *torch.load_state_dict*) with *load_model_data after* the initializing logic.

## Special cases
Sometime there isn't an explicit pipe object as each submodel is loaded separately in the main app. If this is the case, you need to create a dictionary that manually maps all the models.\
For instance :


- for flux derived models: 
```
pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
```
- for mochi: 
```
pipe = { "text_encoder": self.text_encoder, "transformer": self.dit, "vae":self.decoder }
```


Please note that there should be always one model whose Id is 'transformer'. It corresponds to the main image / video model which usually needs to be quantized (this is done on the fly by default when loading the model).

Becareful, lots of models use the T5 XXL as a text encoder. However, quite often their corresponding pipeline configurations point at the official Google T5 XXL repository 
where there is a huge 40GB model to download and load. It is cumbersorme as it is a 32 bits model and contains the decoder part of T5 that is not used. 
I suggest you use instead one of the 16 bits encoder only version available around, for instance:
```
text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=torch.float16)
```

Sometime just providing the pipe won't be sufficient as you will need to change the content of the core model: 
- For instance you may need to disable an existing CPU offload logic that already exists (such as manual calls to move tensors between cuda and the cpu)
- mmpg to tries to fake the device as being "cuda" but sometimes some code won't be fooled and it will create tensors in the cpu device and this may cause some issues.

You are free to use my module for non commercial use as long you give me proper credits. You may contact me on twitter @deepbeepmeep

Thanks to
---------
- Huggingface / accelerate for the hooking examples
- Huggingface / quanto for their very useful quantizer
- gau-nernst for his Pinnig RAM samples