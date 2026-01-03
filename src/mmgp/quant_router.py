import importlib
import inspect

import torch
from optimum.quanto import QModuleMixin, register_qmodule
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype


_QTYPE_QMODULE_CACHE = None
_QMODULE_BASE_ATTRS = None


def _extract_qtypes(handler):
    for obj in vars(handler).values():
        if isinstance(obj, _quanto_qtype):
            yield obj


def _extract_qmodule_classes(handler):
    for obj in vars(handler).values():
        if inspect.isclass(obj) and issubclass(obj, QModuleMixin) and issubclass(obj, torch.nn.Linear):
            if obj is QLinearQuantoRouter:
                continue
            yield obj


def _build_qmodule_cache():
    mapping = {}
    for handler in _load_handlers():
        qmodule_classes = list(_extract_qmodule_classes(handler))
        if len(qmodule_classes) != 1:
            continue
        qmodule_cls = qmodule_classes[0]
        for qt in _extract_qtypes(handler):
            mapping.setdefault(qt, qmodule_cls)
    return mapping


def _get_qmodule_base_attrs():
    global _QMODULE_BASE_ATTRS
    if _QMODULE_BASE_ATTRS is not None:
        return _QMODULE_BASE_ATTRS
    base = torch.nn.Linear(1, 1, bias=True)
    _QMODULE_BASE_ATTRS = set(base.__dict__.keys())
    _QMODULE_BASE_ATTRS.update({
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
    })
    return _QMODULE_BASE_ATTRS


def _get_qmodule_for_qtype(qtype_obj):
    global _QTYPE_QMODULE_CACHE
    if qtype_obj is None:
        return None
    if _QTYPE_QMODULE_CACHE is None or qtype_obj not in _QTYPE_QMODULE_CACHE:
        _QTYPE_QMODULE_CACHE = _build_qmodule_cache()
    return _QTYPE_QMODULE_CACHE.get(qtype_obj)


def _load_with_qmodule(
    module,
    qmodule_cls,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    device = module.weight.device if torch.is_tensor(module.weight) else None
    if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
        weight_dtype = module.weight.dtype
    elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
        weight_dtype = module.bias.dtype
    else:
        weight_dtype = torch.float16
    tmp = qmodule_cls(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=device,
        dtype=weight_dtype,
        weights=module.weight_qtype,
        activations=module.activation_qtype,
        optimizer=module.optimizer,
        quantize_input=True,
    )
    setter = getattr(tmp, "set_default_dtype", None)
    if callable(setter):
        setter(getattr(module, "_router_default_dtype", None) or module.weight.dtype)
    tmp._load_from_state_dict(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    )

    module.weight = tmp.weight
    module.bias = tmp.bias
    module.input_scale = tmp.input_scale
    module.output_scale = tmp.output_scale

    ignore = set(_get_qmodule_base_attrs())
    ignore.update({
        "_quantize_hooks",
        "training",
        "_router_default_dtype",
    })
    for name, value in tmp.__dict__.items():
        if name in ignore:
            continue
        setattr(module, name, value)
    module._router_forward_impl = qmodule_cls.forward


@register_qmodule(torch.nn.Linear)
class QLinearQuantoRouter(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls,
        module,
        weights,
        activations=None,
        optimizer=None,
        device=None,
    ):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._router_default_dtype = dtype

    def set_default_dtype(self, dtype):
        self._router_default_dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        impl = getattr(self, "_router_forward_impl", None)
        if impl is not None:
            return impl(self, input)
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        qmodule_cls = _get_qmodule_for_qtype(self.weight_qtype)
        if qmodule_cls is not None and qmodule_cls is not QLinearQuantoRouter:
            return _load_with_qmodule(
                self, qmodule_cls, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


_HANDLER_MODULES = [
    ".fp8_quanto_bridge"
]
_HANDLER_OBJECTS = []


def register_handler(handler):
    global _QTYPE_QMODULE_CACHE
    if isinstance(handler, str):
        if handler not in _HANDLER_MODULES:
            _HANDLER_MODULES.append(handler)
            _QTYPE_QMODULE_CACHE = None
        return handler
    if handler not in _HANDLER_OBJECTS:
        _HANDLER_OBJECTS.append(handler)
        _QTYPE_QMODULE_CACHE = None
    return handler


def _load_handlers():
    handlers = []
    for mod_path in _HANDLER_MODULES:
        module = importlib.import_module(mod_path, package=__package__)
        if not hasattr(module, "detect") or not hasattr(module, "convert_to_quanto"):
            raise RuntimeError(
                f"Quant handler '{mod_path}' must define detect() and convert_to_quanto() functions."
            )
        handlers.append(module)
    for handler in _HANDLER_OBJECTS:
        if not hasattr(handler, "detect") or not hasattr(handler, "convert_to_quanto"):
            raise RuntimeError(
                "Quant handler object must define detect() and convert_to_quanto() functions."
            )
        handlers.append(handler)
    register_qmodule(torch.nn.Linear)(QLinearQuantoRouter)
    return handlers


def _handler_name(handler):
    return getattr(handler, "HANDLER_NAME", handler.__name__.split(".")[-1])


def detect_safetensors_format(state_dict, verboseLevel=1):
    matches = []
    details = {}
    for handler in _load_handlers():
        result = handler.detect(state_dict, verboseLevel=verboseLevel)
        name = _handler_name(handler)
        details[name] = result
        if result.get("matched", False):
            matches.append(name)
    if len(matches) > 1:
        return {"kind": "mixed", "found": matches, "details": details}
    if len(matches) == 1:
        return {"kind": matches[0], "found": matches, "details": details}
    return {"kind": "none", "found": [], "details": details}


def detect_and_convert(state_dict, default_dtype, verboseLevel=1):
    info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel)
    kind = info.get("kind", "none")
    if kind == "mixed":
        found = info.get("found", [])
        details = info.get("details", {})
        raise RuntimeError(f"Mixed quantization formats detected: {found} details={details}")
    if kind in ("none", "quanto"):
        return {"state_dict": state_dict, "quant_map": {}, "kind": kind, "details": info}
    for handler in _load_handlers():
        if _handler_name(handler) == kind:
            detection = info.get("details", {}).get(kind, {})
            conv = handler.convert_to_quanto(
                state_dict,
                default_dtype=default_dtype,
                verboseLevel=verboseLevel,
                detection=detection,
            )
            conv["kind"] = kind
            conv["details"] = info
            return conv
    raise RuntimeError(f"Unsupported quantization format '{kind}'")


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    remaining = dict(quantization_map or {})
    post_load = []
    for handler in _load_handlers():
        fn = getattr(handler, "apply_pre_quantization", None)
        if fn is None:
            continue
        remaining, hooks = fn(
            model,
            state_dict,
            remaining,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
        )
        if hooks:
            post_load.extend(hooks)
    return remaining, post_load
