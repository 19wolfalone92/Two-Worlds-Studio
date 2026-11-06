"""Static ComfyUI reference-conditioning check, not proof of facial similarity.

An unused LoadImage is never adequate: an enabled IP-Adapter must influence
the model feeding the sampler that produced SaveImage node 9.
"""
from __future__ import annotations

APPLIERS = {"IPAdapterAdvanced", "IPAdapter", "IPAdapterFaceID",
            "IPAdapterFaceIDBatch", "IPAdapterBatch"}
MODEL_PASS = {"ReRoute", "Reroute", "LoraLoader", "LoraLoaderModelOnly",
              "ModelSamplingDiscrete", "ModelSamplingContinuousEDM",
              "FreeU", "FreeU_V2", "DifferentialDiffusion", "PatchModelAddDownscale"}
IMAGE_PASS = {"ReRoute", "Reroute", "ImageScale", "ImageScaleBy",
              "ImageCrop", "ImageResize", "ImagePadForOutpaint"}


def _node(value, graph):
    if (isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and value[0] in graph
        and type(value[1]) is int and value[1] >= 0):
        return value[0]
    return None


def _reaches_reference(value, graph, anchors, seen=frozenset()):
    node_id = _node(value, graph)
    if not node_id or node_id in seen or len(seen) > 12:
        return False
    if node_id in anchors:
        return True
    node = graph[node_id]
    if node.get("class_type") not in IMAGE_PASS:
        return False
    return any(_reaches_reference(v, graph, anchors, seen | {node_id})
               for v in node.get("inputs", {}).values())


def _model_contains_adapter(value, graph, adapters, seen=frozenset()):
    node_id = _node(value, graph)
    if not node_id or node_id in seen or len(seen) > 15:
        return False
    if node_id in adapters:
        return True
    node = graph[node_id]
    if node.get("class_type") not in MODEL_PASS:
        return False
    return any(_model_contains_adapter(v, graph, adapters, seen | {node_id})
               for k, v in node.get("inputs", {}).items()
               if k in {"model", "model1", "model2"})


def audit_reference_conditioning(graph: dict) -> dict:
    if not isinstance(graph, dict):
        raise ValueError("Expected ComfyUI API workflow object")
    anchor = {id_ for id_, node in graph.items()
              if isinstance(node, dict) and node.get("class_type") == "LoadImage"
              and node.get("inputs", {}).get("image") == "__REFERENCE_IMAGE__"}
    if len(anchor) != 1:
        raise ValueError("Expected exactly one reference LoadImage placeholder")
    applying = set()
    for id_, node in graph.items():
        if not isinstance(node, dict) or node.get("class_type") not in APPLIERS:
            continue
        params = node.get("inputs", {})
        weight = params.get("weight")
        if (type(weight) in (int, float) and .01 <= weight <= 2
            and _node(params.get("model"), graph)
            and _node(params.get("ipadapter"), graph)
            and _reaches_reference(params.get("image"), graph, anchor)):
            applying.add(id_)
    if not applying:
        raise ValueError("Reference is not connected to an enabled IP-Adapter model")
    saved = graph.get("9", {})
    if saved.get("class_type") != "SaveImage":
        raise ValueError("SaveImage node 9 missing")
    decoder_id = _node(saved.get("inputs", {}).get("images"), graph)
    if not decoder_id or graph[decoder_id].get("class_type") != "VAEDecode":
        raise ValueError("SaveImage is not fed by VAEDecode")
    sampler_id = _node(graph[decoder_id].get("inputs", {}).get("samples"), graph)
    if not sampler_id or graph[sampler_id].get("class_type") not in {"KSampler", "KSamplerAdvanced"}:
        raise ValueError("Saved image is not produced by sampler")
    if not _model_contains_adapter(graph[sampler_id].get("inputs", {}).get("model"), graph, applying):
        raise ValueError("Saved image sampler does not use the reference-conditioned model")
    return {"loader_node": next(iter(anchor)), "adapter_nodes": sorted(applying),
            "sampler_node": sampler_id, "topological_reference_path": True,
            "face_similarity_verified": False, "model_weights_verified": False,
            "human_identity_review_required": True}
