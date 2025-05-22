import torch
import torch.nn as nn
import torch.nn.functional as F


class ICAModule(nn.Module):
    def __init__(self, c, d, h, l):
        super(ICAModule, self).__init__()
        self.linear_projection = nn.Linear(c, d)
        self.h = h
        self.l = l
        # self.x_t = nn.Parameter(torch.randn(d))  # KT token
        self.x_r = nn.Parameter(torch.randn(d))  # KR token

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, Y, x_t):
        X_p = self.linear_projection(Y)
        batch_size = X_p.size(0)
        x_t = x_t.expand(batch_size, -1)  # (batch_size, d)
        x_r = self.x_r.unsqueeze(0).expand(batch_size, -1)  # (batch_size, d)

        Q = self.W_q(x_t).unsqueeze(1)  # (batch_size, 1, d)
        K = self.W_k(torch.cat([x_r.unsqueeze(1), X_p], dim=1))  # (batch_size, L+1, d)
        V = self.W_v(torch.cat([x_r.unsqueeze(1), X_p], dim=1))  # (batch_size, L+1, d)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.l ** 0.5)  # (batch_size, 1, L+1)
        attention_probs = F.softmax(attention_scores, dim=-1)
        z = torch.matmul(attention_probs, V).squeeze(1)  # (batch_size, d)

        e_1 = x_t + self.norm(z)
        e = e_1 + self.mlp(self.norm(e_1))

        return e


class ICABasedMLCIL(nn.Module):
    def __init__(self, c, d, h, l, num_classes_list):
        super(ICABasedMLCIL, self).__init__()
        self.x_t = nn.Parameter(torch.randn(d))  # Shared x_t token
        self.ica_modules = nn.ModuleList([ICAModule(c, d, h, l) for _ in num_classes_list])
        self.classification_heads = nn.ModuleList([nn.Linear(d, num_classes) for num_classes in num_classes_list])

    def forward(self, x, session_idx):
        x_t = self.x_t.unsqueeze(0)  # Ensure x_t has the correct shape
        logits_list = []
        for i in range(session_idx + 1):
            X_p = self.ica_modules[session_idx](x, x_t)
            x_t = X_p  # Update x_t for the next task
            logits = self.classification_heads[i](X_p)
            logits_list.append(logits)

        return torch.cat(logits_list, dim=1)


