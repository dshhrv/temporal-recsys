import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, use_item_bias=True):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_item_bias = use_item_bias

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        if use_item_bias:
            self.item_bias = nn.Parameter(torch.zeros(num_items))
        else:
            self.register_parameter("item_bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias)

    def ego_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def propagate(self, norm_adj):
        embeddings = self.ego_embeddings()
        layer_embeddings = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(norm_adj, embeddings)
            layer_embeddings.append(embeddings)

        embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        user_embeddings, item_embeddings = torch.split(embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def full_sort_logits(self, user_ids, user_embeddings, item_embeddings):
        logits = user_embeddings[user_ids] @ item_embeddings.t()

        if self.item_bias is not None:
            logits = logits + self.item_bias.unsqueeze(0)

        return logits

    def ce_loss(self, logits, item_ids):
        return F.cross_entropy(logits, item_ids)

    def embedding_regularization(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        return 0.5 * (user_embeddings.pow(2).sum(dim=1) + item_embeddings.pow(2).sum(dim=1)).mean()
