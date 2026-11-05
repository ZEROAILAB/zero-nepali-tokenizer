import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ZERO 100M CONFIGURATION
# ============================================================

VOCAB_SIZE = 10956
CONTEXT_LENGTH = 1024

N_LAYERS = 12
N_HEADS = 12

EMBED_DIM = 768
FFN_DIM = 3072


# ============================================================
# SELF ATTENTION
# ============================================================

class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        assert EMBED_DIM % N_HEADS == 0

        self.head_dim = EMBED_DIM // N_HEADS

        self.qkv = nn.Linear(
            EMBED_DIM,
            EMBED_DIM * 3
        )

        self.output = nn.Linear(
            EMBED_DIM,
            EMBED_DIM
        )

    def forward(self, x):

        batch_size, seq_length, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_length,
            N_HEADS,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_length,
            N_HEADS,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_length,
            N_HEADS,
            self.head_dim
        ).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        attention = attention.transpose(
            1, 2
        ).contiguous()

        attention = attention.view(
            batch_size,
            seq_length,
            EMBED_DIM
        )

        return self.output(attention)


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            EMBED_DIM,
            FFN_DIM
        )

        self.fc2 = nn.Linear(
            FFN_DIM,
            EMBED_DIM
        )

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.fc2(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            EMBED_DIM
        )

        self.attention = SelfAttention()

        self.norm2 = nn.LayerNorm(
            EMBED_DIM
        )

        self.mlp = MLP()

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# ZERO TRANSFORMER
# ============================================================

class ZEROTransformer(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            CONTEXT_LENGTH,
            EMBED_DIM
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(N_LAYERS)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            EMBED_DIM
        )

        # Output head
        self.output_head = nn.Linear(
            EMBED_DIM,
            vocab_size,
            bias=False
        )

        # Initialize everything once
        self.apply(self._init_weights)

        # Scale residual projections
        residual_scale = 1.0 / math.sqrt(
            2 * N_LAYERS
        )

        for block in self.blocks:

            nn.init.normal_(
                block.attention.output.weight,
                mean=0.0,
                std=0.02 * residual_scale
            )

            nn.init.normal_(
                block.mlp.fc2.weight,
                mean=0.0,
                std=0.02 * residual_scale
            )

        # Tie output weights to token embeddings
        self.output_head.weight = (
            self.token_embedding.weight
        )

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        batch_size, seq_length = (
            input_ids.shape
        )

        assert seq_length <= CONTEXT_LENGTH

        # Position IDs
        positions = torch.arange(
            seq_length,
            device=input_ids.device
        )

        # Token embeddings
        token_embeddings = (
            self.token_embedding(input_ids)
        )

        # Position embeddings
        position_embeddings = (
            self.position_embedding(positions)
        )

        # Combine embeddings
        x = (
            token_embeddings
            + position_embeddings
        )

        # Transformer
        for block in self.blocks:

            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.output_head(x)

        return logits


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("====================================")
    print("        ZERO 100M MODEL")
    print("====================================")

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print("Device:", device)

    model = ZEROTransformer()

    model = model.to(device)

    parameters = count_parameters(model)

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Parameters (millions):",
        f"{parameters / 1_000_000:.2f}M"
    )

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (2, 32),
        device=device
    )

    print(
        "Input shape:",
        input_ids.shape
    )

    with torch.no_grad():

        logits = model(input_ids)

    print(
        "Output shape:",
        logits.shape
    )

    print("------------------------------------")
    print("ZERO TRANSFORMER TEST COMPLETE")
    print("------------------------------------")
