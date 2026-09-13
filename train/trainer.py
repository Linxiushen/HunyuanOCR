from transformers import Trainer
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLModel,
    HunYuanVLVisionTransformer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


# NOTE: HunyuanOCR packed training uses the model's NATIVE packing support.
# The old flash_attention_2 monkey-patch (hunyuanvl_forward / apply_rotary_pos_emb_xdrope /
# replace_hunyuanocr_attention_class) targeted a `HunYuanVLAttention` class that no longer
# exists in transformers>=5.15 (superseded by `HunYuanVLDenseV1Attention`, which already
# applies the correct multimodal RoPE). Packing is instead achieved by feeding per-sample
# restarted 3D position_ids plus a block-diagonal causal mask (see PackedVLDataCollator),
# which the native forward consumes directly. No attention patching is required.


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.layers):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.perceive.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}")
    print(f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}")
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(param.requires_grad for param in self.embed_tokens.parameters())
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}")
    print(f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}")


def create_optimizer(self):
    """Build the optimizer, optionally giving the projector / vision tower their own lr."""
    opt_model = self.model

    if self.optimizer is None:
        decay = {name for name in self.get_decay_parameter_names(opt_model) if "bias" not in name}
        named = list(opt_model.named_parameters())
        wd = self.args.weight_decay
        mm_lr = self.args.mm_projector_lr
        vt_lr = self.args.vision_tower_lr

        def group(keep, weight_decay, lr=None):
            g = {"params": [p for n, p in named if keep(n) and p.requires_grad], "weight_decay": weight_decay}
            if lr is not None:
                g["lr"] = lr
            return g

        if mm_lr:
            projector = {n for n, _ in named if "perceive" in n}
            if vt_lr:
                vision = {n for n, _ in named if "vit" in n}
                optimizer_grouped_parameters = [
                    group(lambda n: n in decay and n not in projector and n not in vision, wd),
                    group(lambda n: n in decay and n not in projector and n in vision, wd, vt_lr),
                    group(lambda n: n not in decay and n not in projector and n not in vision, 0.0),
                    group(lambda n: n not in decay and n not in projector and n in vision, 0.0, vt_lr),
                    group(lambda n: n in decay and n in projector, wd, mm_lr),
                    group(lambda n: n not in decay and n in projector, 0.0, mm_lr),
                ]
            else:
                optimizer_grouped_parameters = [
                    group(lambda n: n in decay and n not in projector, wd),
                    group(lambda n: n not in decay and n not in projector, 0.0),
                    group(lambda n: n in decay and n in projector, wd, mm_lr),
                    group(lambda n: n not in decay and n in projector, 0.0, mm_lr),
                ]
        else:
            optimizer_grouped_parameters = [
                group(lambda n: n in decay, wd),
                group(lambda n: n not in decay, 0.0),
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

HunYuanVLVisionTransformer.print_trainable_parameters = print_trainable_parameters_visual
HunYuanVLModel.print_trainable_parameters = print_trainable_parameters
