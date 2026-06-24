import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.linear = nn.Linear(1, time_dim)

    def forward(self, delta_t):
        return torch.cos(self.linear(delta_t.float().unsqueeze(-1)))


class TemporalGraphAttention(nn.Module):
    def __init__(self, node_dim, time_dim, edge_dim=1, num_heads=1, dropout=0.1):
        super().__init__()
        self.edge_dim = edge_dim
        self.attention_dim = node_dim + edge_dim + time_dim
        self.attention = nn.MultiheadAttention(self.attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.merge = nn.Sequential(nn.Linear(node_dim + self.attention_dim, node_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(node_dim, node_dim))

    def forward(self, node_states, node_time, neighbor_features, mask):
        has_neighbors = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~has_neighbors, 0] = True
        safe_features = neighbor_features.clone()
        safe_features[~has_neighbors, 0] = 0.0
        query_edge_features = node_states.new_zeros(node_states.size(0), self.edge_dim)
        query = torch.cat([node_states, query_edge_features, node_time], dim=-1).unsqueeze(1)
        attended, _ = self.attention(query, safe_features, safe_features, key_padding_mask=~safe_mask, need_weights=False)
        attended = attended.squeeze(1) * has_neighbors.unsqueeze(-1)
        return self.merge(torch.cat([node_states, attended], dim=-1))


class TGN(nn.Module):
    def __init__(self, num_users, num_items, node_dim=31, memory_dim=31, time_dim=100, edge_dim=1, num_neighbors=10, num_heads=1, dropout=0.1, use_item_bias=True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.edge_dim = edge_dim
        self.num_neighbors = num_neighbors
        self.use_item_bias = use_item_bias
        self.node_embedding = nn.Embedding(self.num_nodes, node_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.memory_to_node = nn.Linear(memory_dim, node_dim, bias=False)
        self.memory_updater = nn.GRUCell(memory_dim * 2 + time_dim + edge_dim, memory_dim)
        self.graph_attention = TemporalGraphAttention(node_dim, time_dim, edge_dim, num_heads, dropout)

        if use_item_bias:
            self.item_bias = nn.Parameter(torch.zeros(num_items))
        else:
            self.register_parameter("item_bias", None)

        self.register_buffer("memory", torch.zeros(self.num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(self.num_nodes))
        self.register_buffer("all_item_ids", torch.arange(num_users, num_users + num_items))
        self.register_buffer("history_node_ids", torch.zeros(self.num_nodes, num_neighbors, dtype=torch.long))
        self.register_buffer("history_times", torch.zeros(self.num_nodes, num_neighbors))
        self.register_buffer("history_edge_features", torch.zeros(self.num_nodes, num_neighbors, edge_dim))
        self.register_buffer("history_mask", torch.zeros(self.num_nodes, num_neighbors, dtype=torch.bool))
        self.pending = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.memory_to_node.weight)

        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias)

    def reset_state(self):
        self.memory = torch.zeros_like(self.memory)
        self.last_update = torch.zeros_like(self.last_update)
        self.history_node_ids = torch.zeros_like(self.history_node_ids)
        self.history_times = torch.zeros_like(self.history_times)
        self.history_edge_features = torch.zeros_like(self.history_edge_features)
        self.history_mask = torch.zeros_like(self.history_mask)
        self.pending = {}

    def detach_state(self):
        self.memory = self.memory.detach()
        self.last_update = self.last_update.detach()

    def node_states(self, node_ids):
        return self.node_embedding(node_ids) + self.memory_to_node(self.memory[node_ids])

    def get_neighbors(self, node_ids, timestamps):
        neighbor_ids = self.history_node_ids[node_ids]
        neighbor_times = self.history_times[node_ids]
        neighbor_edge_features = self.history_edge_features[node_ids]
        mask = self.history_mask[node_ids] & (neighbor_times < timestamps.float().unsqueeze(1))
        delta_t = (timestamps.float().unsqueeze(1) - neighbor_times).clamp_min(0.0)
        return neighbor_ids, delta_t, neighbor_edge_features, mask

    def encode_nodes(self, node_ids, timestamps):
        node_states = self.node_states(node_ids)
        neighbor_ids, neighbor_delta_t, neighbor_edge_features, mask = self.get_neighbors(node_ids, timestamps)
        neighbor_states = self.node_states(neighbor_ids)
        neighbor_time = self.time_encoder(neighbor_delta_t)
        neighbor_features = torch.cat([neighbor_states, neighbor_edge_features, neighbor_time], dim=-1)
        zero_time = self.time_encoder(torch.zeros_like(timestamps))
        return self.graph_attention(node_states, zero_time, neighbor_features, mask)

    def full_ce_logits(self, user_ids, timestamps, item_chunk_size=256):
        batch_size = user_ids.size(0)
        user_embeddings = self.encode_nodes(user_ids, timestamps)
        logits = []

        for item_chunk in self.all_item_ids.split(item_chunk_size):
            chunk_size = item_chunk.size(0)
            candidate_ids = item_chunk.unsqueeze(0).expand(batch_size, chunk_size).reshape(-1)
            candidate_times = timestamps.unsqueeze(1).expand(batch_size, chunk_size).reshape(-1)
            candidate_embeddings = self.encode_nodes(candidate_ids, candidate_times).reshape(batch_size, chunk_size, -1)
            chunk_logits = (user_embeddings.unsqueeze(1) * candidate_embeddings).sum(dim=-1)

            if self.item_bias is not None:
                chunk_logits = chunk_logits + self.item_bias[(item_chunk - self.num_users).long()].unsqueeze(0)

            logits.append(chunk_logits)

        return torch.cat(logits, dim=1)

    def forward(self, user_ids, timestamps, item_chunk_size=256):
        return self.full_ce_logits(user_ids, timestamps, item_chunk_size)

    def normalize_edge_features(self, edge_features, reference):
        if edge_features is None:
            return reference.new_zeros(reference.size(0), self.edge_dim)

        edge_features = edge_features.to(device=reference.device, dtype=reference.dtype)

        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        if edge_features.size(-1) != self.edge_dim:
            raise ValueError(f"Expected edge_features with last dimension {self.edge_dim}, got {edge_features.size(-1)}")

        return edge_features

    def make_message(self, self_memory, other_memory, delta_t, edge_features):
        return torch.cat([self_memory, other_memory, self.time_encoder(delta_t), edge_features], dim=-1)

    def add_pending_message(self, node_id, self_memory, other_memory, delta_time, edge_feature, timestamp):
        self.pending.setdefault(node_id, []).append((self_memory, other_memory, delta_time, edge_feature, timestamp))

    def flush_pending_messages(self):
        if not self.pending:
            return

        node_ids = []
        messages = []
        timestamps = []

        for node_id, node_messages in self.pending.items():
            self_memories = torch.stack([value[0] for value in node_messages])
            other_memories = torch.stack([value[1] for value in node_messages])
            delta_times = torch.stack([value[2] for value in node_messages])
            edge_features = torch.stack([value[3] for value in node_messages])
            node_timestamps = [value[4] for value in node_messages]
            node_ids.append(node_id)
            messages.append(self.make_message(self_memories, other_memories, delta_times, edge_features).mean(dim=0))
            timestamps.append(max(node_timestamps))

        node_ids = torch.tensor(node_ids, dtype=torch.long, device=self.memory.device)
        messages = torch.stack(messages)
        timestamps = torch.tensor(timestamps, dtype=torch.float32, device=self.memory.device)
        updated_memory = self.memory_updater(messages, self.memory[node_ids])
        self.memory = self.memory.index_copy(0, node_ids, updated_memory)
        self.last_update = self.last_update.index_copy(0, node_ids, timestamps)
        self.pending = {}

    def append_history(self, node_id, neighbor_id, timestamp, edge_feature):
        self.history_node_ids[node_id, :-1] = self.history_node_ids[node_id, 1:].clone()
        self.history_times[node_id, :-1] = self.history_times[node_id, 1:].clone()
        self.history_edge_features[node_id, :-1] = self.history_edge_features[node_id, 1:].clone()
        self.history_mask[node_id, :-1] = self.history_mask[node_id, 1:].clone()
        self.history_node_ids[node_id, -1] = neighbor_id
        self.history_times[node_id, -1] = timestamp
        self.history_edge_features[node_id, -1] = edge_feature
        self.history_mask[node_id, -1] = True

    def store_batch(self, user_ids, item_ids, timestamps, edge_features=None):
        with torch.no_grad():
            item_node_ids = item_ids + self.num_users
            edge_features = self.normalize_edge_features(edge_features, timestamps.float()).detach().clone()
            user_memory = self.memory[user_ids].detach().clone()
            item_memory = self.memory[item_node_ids].detach().clone()
            user_delta_t = (timestamps.float() - self.last_update[user_ids]).clamp_min(0.0).detach().clone()
            item_delta_t = (timestamps.float() - self.last_update[item_node_ids]).clamp_min(0.0).detach().clone()
            users = user_ids.detach().cpu().tolist()
            items = item_node_ids.detach().cpu().tolist()
            times = timestamps.detach().cpu().tolist()

            for row, (user_id, item_node_id, timestamp) in enumerate(zip(users, items, times)):
                edge_feature = edge_features[row]
                self.add_pending_message(user_id, user_memory[row], item_memory[row], user_delta_t[row], edge_feature, float(timestamp))
                self.add_pending_message(item_node_id, item_memory[row], user_memory[row], item_delta_t[row], edge_feature, float(timestamp))
                self.append_history(user_id, item_node_id, float(timestamp), edge_feature)
                self.append_history(item_node_id, user_id, float(timestamp), edge_feature)
