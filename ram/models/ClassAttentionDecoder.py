import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAttentionDecoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, num_heads, num_blocks):
        super(ClassAttentionDecoder, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.class_tokens = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, embedding_dim))

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads) for _ in range(num_blocks)
        ])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.ReLU(),
                nn.Linear(embedding_dim * 4, embedding_dim),
            ) for _ in range(num_blocks)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])
        # self.final_proj = nn.Linear(embedding_dim, 1)

    def forward(self, feature_embeddings):
        batch_size, tokens, embedding_dim = feature_embeddings.shape
        feature_embeddings = feature_embeddings.view(batch_size, tokens, embedding_dim)
        feature_embeddings = feature_embeddings.permute(1, 0, 2)  # Reshape for attention

        queries = self.class_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        queries = queries + self.pos_embedding

        for i in range(self.num_blocks):
            queries = self.layer_norms[i](queries)
            attn_output, _ = self.attention_layers[i](queries, feature_embeddings, feature_embeddings)
            queries = queries + attn_output
            queries = queries + self.mlp_layers[i](queries)

        queries = queries[:, :, 0].transpose(0, 1)
        # queries = self.final_proj(queries).squeeze(-1).transpose(0, 1)

        return queries



