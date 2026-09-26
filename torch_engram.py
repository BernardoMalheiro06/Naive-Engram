import torch
from math import gcd

class EmbeddingLookup(torch.nn.Module):
    def __init__(self, hash_vocab_size, embed_size, n_tables=1, engram_size=2, pad_token=0, device="cpu"):
        super().__init__()
        assert embed_size % n_tables == 0, "embed_size must be divisible by n_tables"
        self.tables = torch.nn.ModuleList([
            torch.nn.Embedding(
                num_embeddings=hash_vocab_size, 
                embedding_dim=embed_size // n_tables,
                device=device,
            )
            for _ in range(n_tables)
        ])
        self.engram_size = engram_size
        self.hash_vocab_size = hash_vocab_size
        self.n_tables = n_tables
        self.embed_size = embed_size
        self.pad_token = pad_token
        self.register_buffer("multipliers", self.generate_multipliers(device))
    
    def generate_multipliers(self, device):
        multipliers = []
        for _ in range(self.n_tables):
            table_multipliers = []
            while len(table_multipliers) < self.engram_size:
                m = torch.randint(
                    self.hash_vocab_size // 2,
                    self.hash_vocab_size,
                    (1,)
                ).item() | 1
                if m not in table_multipliers and gcd(m, self.hash_vocab_size) == 1:
                    table_multipliers.append(m)
            multipliers.append(table_multipliers)
        return torch.tensor(multipliers, dtype=torch.long, device=device)
    
    def hash(self, x: torch.Tensor):
        x = torch.nn.functional.pad(x, (self.engram_size - 1, 0), value=self.pad_token)
        engrams = x.unfold(dimension=1, size=self.engram_size, step=1)
        engrams = engrams[:, :, None, :] * self.multipliers[None, None, :, :]
        xor_hashes = engrams[..., 0].clone()
        for i in range(1, engrams.size()[-1]):
            xor_hashes = torch.bitwise_xor(xor_hashes, engrams[..., i])
        hashes = xor_hashes % self.hash_vocab_size
        return hashes.permute(2, 0, 1)
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.long
        assert x.dim() == 2

        hashes = self.hash(x)
        embeddings = [self.tables[i](hashes[i]) for i in range(self.n_tables)]
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings



class ContextAwareGating(torch.nn.Module):
    def __init__(self, embed_size, n_heads, device="cpu"):
        super().__init__()
        assert embed_size % n_heads == 0
        self.n_heads = n_heads
        self.embed_size = embed_size
        self.W_k = torch.nn.Linear(self.embed_size // self.n_heads, self.embed_size // self.n_heads, bias=False, device=device)
        self.W_v = torch.nn.Linear(self.embed_size // self.n_heads, self.embed_size // self.n_heads, bias=False, device=device)

    def forward(self, x, retrieved_embeds):
        assert x.shape == retrieved_embeds.shape
        B, S, D = x.shape
        H, D_head = self.n_heads, self.embed_size // self.n_heads

        Q = x.view(B, S, H, D_head).transpose(1, 2)
        retrieved_embeds = retrieved_embeds.view(B, S, H, D_head).transpose(1, 2)

        K = self.W_k(retrieved_embeds)
        V = self.W_v(retrieved_embeds)

        Q = torch.nn.functional.rms_norm(Q, (D_head,))
        K = torch.nn.functional.rms_norm(K, (D_head,))

        scores = (Q * K).sum(dim=-1).unsqueeze(-1) / (D_head ** 0.5)

        return (scores * V).transpose(1, 2).contiguous().view(B, S, D)



class ShortConv(torch.nn.Module):
    def __init__(self, embed_size, kernel_size, engram_size, device="cpu"):
        super().__init__()
        self.conv = torch.nn.Conv1d(embed_size, embed_size, kernel_size=kernel_size, dilation=engram_size, device=device)

    def forward(self, x):
        _, _, D = x.shape
        out = torch.nn.functional.rms_norm(x, (D,)).transpose(1, 2)
        pad = self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        out = torch.nn.functional.pad(out, (pad, 0))
        return torch.nn.functional.silu(self.conv(out).transpose(1, 2)) + x



class Engram(torch.nn.Module):
    def __init__(self, embed_size, n_heads, kernel_size, engram_size, device="cpu"):
        super().__init__()
        self.context_aware_gating = ContextAwareGating(embed_size=embed_size, n_heads=n_heads, device=device)
        self.short_conv = ShortConv(embed_size=embed_size, kernel_size=kernel_size, engram_size=engram_size, device=device)

    def forward(self, x, retrieved_embeds):
        v = self.context_aware_gating(x, retrieved_embeds)
        return x + self.short_conv(v) + v