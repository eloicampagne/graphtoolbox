import torch
from torch_geometric.nn import GATConv, TransformerConv, GCNConv, SAGEConv
from inspect import signature
from einops import rearrange
from torch_geometric.nn.models.deepgcn import DeepGCNLayer

import torch.nn as nn
from torch_geometric.nn import (
    APPNP, ChebConv, GatedGraphConv, MixHopConv, GCN2Conv,
    DirGNNConv, GravNetConv, NNConv, EdgeConv, DNAConv, SignedConv
)
from torch_geometric.nn.conv import WLConv

def _mlp(ch_in, ch_out, hidden=None, act=nn.ReLU, last_act=False):
    hidden = hidden or max(ch_in, ch_out)
    layers = [nn.Linear(ch_in, hidden), act()]
    layers += [nn.Linear(hidden, ch_out)]
    if last_act:
        layers += [act()]
    return nn.Sequential(*layers)

class ConvAdapter(nn.Module):
    def __init__(self, conv_class, in_dim, hidden_channels, heads=1, base_kwargs=None):
        super().__init__()
        base_kwargs = base_kwargs or {}
        self.conv_class = conv_class
        self.in_dim = in_dim
        self.hidden_channels = hidden_channels
        self.heads = heads
        name = conv_class.__name__

        try:
            ctor_params = set(signature(conv_class).parameters.keys())
        except Exception:
            ctor_params = set()

        kwargs = {}
        supports_heads = "heads" in ctor_params
        supports_concat = "concat" in ctor_params

        # FusedGATConv uses *args/**kwargs and inherits GATConv; inject dims manually
        if name == "FusedGATConv":
            h = self.heads
            kwargs["in_channels"] = in_dim
            if h > 0 and (in_dim % h) == 0:
                kwargs["out_channels"] = in_dim // h
                kwargs["concat"] = True
            else:
                kwargs["out_channels"] = in_dim
                kwargs["concat"] = False
            kwargs["heads"] = h
            kwargs["add_self_loops"] = False  # FusedGATConv does not support self-loops

        # --- dimension policy ---
        if "channels" in ctor_params:
            kwargs["channels"] = in_dim
        if "in_channels" in ctor_params:
            kwargs["in_channels"] = in_dim
        if "out_channels" in ctor_params:
            if supports_heads and supports_concat and base_kwargs.get("concat", True):
                if self.heads == 0 or (in_dim % self.heads) != 0:
                    kwargs["out_channels"] = in_dim
                    kwargs["concat"] = False
                else:
                    kwargs["out_channels"] = in_dim // self.heads
                    kwargs["concat"] = True
            else:
                # keep feature size unchanged across layers
                if name == "MixHopConv":
                    _powers = base_kwargs.get("powers", [0, 1, 2])
                    _n = len(_powers)
                    kwargs["out_channels"] = max(1, in_dim // _n)
                elif name == "SignedConv":
                    # first_aggr=True: output is 2*out_channels; halve to land on in_dim
                    kwargs["out_channels"] = max(1, in_dim // 2)
                else:
                    kwargs["out_channels"] = in_dim
        if supports_heads:
            kwargs["heads"] = self.heads
        if "cached" in ctor_params:
            kwargs["cached"] = False
        if "add_self_loops" in ctor_params and name != "FusedGATConv":
            kwargs["add_self_loops"] = base_kwargs.get("add_self_loops", True)
        if "edge_dim" in ctor_params:
            kwargs["edge_dim"] = base_kwargs.get("edge_dim", 1)

        # --- specific layer defaults ---
        if conv_class is APPNP:
            kwargs.update({"K": 4, "alpha": 0.9})
        if conv_class is ChebConv and "K" in ctor_params:
            kwargs["K"] = 3
        if conv_class is GatedGraphConv and "num_layers" in ctor_params:
            kwargs["num_layers"] = 2
            if "out_channels" in ctor_params:
                kwargs["out_channels"] = in_dim
        if conv_class is MixHopConv:
            _powers = base_kwargs.get("powers", [0, 1, 2])
            kwargs.setdefault("powers", _powers)
        if conv_class is GCN2Conv and "alpha" in ctor_params:
            kwargs.setdefault("alpha", 0.1)
        if name == "SSGConv" and "alpha" in ctor_params:
            kwargs.setdefault("alpha", 0.1)
        if name == "CGConv":
            # CGConv requires 'dim' to match edge_attr size (not node feature size)
            kwargs["dim"] = base_kwargs.get("dim", base_kwargs.get("edge_dim", 1))
        if name == "GPSConv":
            # provide an instantiated base message-passing conv
            kwargs["conv"] = GCNConv(in_dim, in_dim)
        if name == "PANConv":
            kwargs["filter_size"] = 2
        if name == "PDNConv":
            kwargs["hidden_channels"] = in_dim
        if conv_class is DirGNNConv and "conv" in ctor_params:
            kwargs["conv"] = GCNConv(in_dim, in_dim)  # must be an instance
        if conv_class is GravNetConv:
            kwargs.setdefault("space_dimensions", min(4, max(2, in_dim)))
            kwargs.setdefault("propagate_dimensions", min(in_dim, 16))
            kwargs.setdefault("k", 3)

        # --- relational convolutions: default to 1 relation ---
        if name in ("RGCNConv", "FastRGCNConv", "RGATConv") and "num_relations" in ctor_params:
            kwargs.setdefault("num_relations", 1)

        # --- convolutions requiring extra constructor arguments ---
        if name == "DynamicEdgeConv" and "nn" in ctor_params:
            kwargs["nn"] = _mlp(2 * in_dim, in_dim, hidden=in_dim)
            kwargs.setdefault("k", 6)
        if name == "GMMConv" and "dim" in ctor_params:
            kwargs.setdefault("dim", base_kwargs.get("edge_dim", 1))
            kwargs.setdefault("kernel_size", 5)
        if name == "GeneralConv" and "in_edge_channels" in ctor_params:
            kwargs.setdefault("in_edge_channels", base_kwargs.get("edge_dim", 1))
        if name == "PNAConv" and "aggregators" in ctor_params:
            kwargs.setdefault("aggregators", ["mean", "max", "sum"])
            kwargs.setdefault("scalers", ["identity"])
            if "deg" not in kwargs and "deg" not in base_kwargs:
                kwargs["deg"] = torch.ones(10, dtype=torch.long)
        if name == "SplineConv" and "dim" in ctor_params:
            kwargs.setdefault("dim", base_kwargs.get("edge_dim", 1))
            kwargs.setdefault("kernel_size", 5)
        if name == "XConv" and "dim" in ctor_params:
            kwargs.setdefault("dim", 3)
            kwargs.setdefault("kernel_size", 5)
            kwargs.setdefault("hidden_channels", in_dim)

        # constructor-MLPs
        if conv_class is NNConv:
            oc = kwargs.get("out_channels", in_dim)
            w_out = in_dim * oc
            nn_edge = _mlp(base_kwargs.get("edge_dim", 1), w_out, hidden=in_dim)
            kwargs["nn"] = nn_edge
        if conv_class is EdgeConv and "nn" in ctor_params:
            kwargs["nn"] = _mlp(2 * in_dim, in_dim, hidden=in_dim)

        # For GIN/GINE provide MLPs
        if name in ("GINConv", "GINEConv") and "nn" in ctor_params:
            kwargs["nn"] = _mlp(in_dim, kwargs.get("out_channels", in_dim), hidden=in_dim)
        if name == "SignedConv" and "first_aggr" in ctor_params:
            kwargs.setdefault("first_aggr", True)

        # instantiate convolution
        self.conv = conv_class(**kwargs)
        self.capture_attention = False
        self.last_attention = None

        # Convolutions whose output dim differs from in_dim: project back
        if name == "MixHopConv":
            _powers = kwargs.get("powers", base_kwargs.get("powers", [0, 1, 2]))
            _n = len(_powers)
            _oc = kwargs.get("out_channels", max(1, in_dim // _n))
            actual_out = _n * _oc
            self.proj = nn.Linear(actual_out, in_dim) if actual_out != in_dim else None
        elif name == "SignedConv":
            # first_aggr=True: actual output is 2*out_channels
            _oc = kwargs.get("out_channels", max(1, in_dim // 2))
            actual_out = 2 * _oc
            self.proj = nn.Linear(actual_out, in_dim) if actual_out != in_dim else None
        else:
            self.proj = None

        try:
            self._forward_params = set(signature(self.conv.forward).parameters.keys())
        except Exception:
            self._forward_params = set()

    def forward(self, x, edge_index, x0=None, **tensors):
        # DNAConv expects x: [N, L, C]; fallback to L=1 if given [N, C]
        if isinstance(self.conv, DNAConv) and x.dim() == 2:
            x = x.unsqueeze(1)

        # WLConv requires 1D integer labels; convert float features to one-hot then back
        if isinstance(self.conv, WLConv):
            idx = x.argmax(dim=-1) if x.dim() > 1 else x.long()
            x_one_hot = torch.zeros_like(x if x.dim() > 1 else x.unsqueeze(-1).expand(-1, self.in_dim))
            x_one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
            out_long = self.conv(x_one_hot, edge_index)          # [N] long hash
            out_float = torch.zeros(out_long.size(0), self.in_dim, device=x.device)
            out_float.scatter_(1, (out_long % self.in_dim).unsqueeze(1), 1.0)
            return out_float

        call = {"x": x}
        if "edge_index" in self._forward_params:
            call["edge_index"] = edge_index
        elif "hyperedge_index" in self._forward_params:
            call["hyperedge_index"] = edge_index

        # common optional tensors
        for key in ("edge_weight", "edge_attr", "pos", "batch", "edge_type", "pseudo", "normal"):
            if key in self._forward_params and tensors.get(key) is not None:
                call[key] = tensors[key]

        # FAConv with normalize=True internally runs gcn_norm and asserts edge_weight is None
        if type(self.conv).__name__ == "FAConv" and getattr(self.conv, "normalize", False):
            call.pop("edge_weight", None)

        E = edge_index.size(1)
        dev = x.device

        # dummies where required
        if "pos" in self._forward_params and "pos" not in call:
            call["pos"] = torch.zeros(x.size(0), 3, device=dev)
        if "normal" in self._forward_params and "normal" not in call:
            call["normal"] = torch.zeros(x.size(0), 3, device=dev)
        if "edge_type" in self._forward_params and "edge_type" not in call:
            call["edge_type"] = torch.zeros(E, dtype=torch.long, device=dev)
        if "pseudo" in self._forward_params and "pseudo" not in call:
            dim = getattr(self.conv, "dim", 3)
            call["pseudo"] = torch.zeros(E, dim, device=dev)
        if "x_0" in self._forward_params and x0 is not None:
            call["x_0"] = x0

        # Some convs hard-require edge_attr; provide a safe default
        # Note: SignedConv is excluded — it uses pos/neg_edge_index, not edge_attr
        needs_edge_attr = isinstance(self.conv, (TransformerConv, NNConv)) or \
                          ("edge_attr" in self._forward_params and "edge_attr" not in call)
        if needs_edge_attr and "edge_attr" not in call:
            ed = getattr(self.conv, "edge_dim", 1)
            call["edge_attr"] = torch.ones(E, ed, device=dev)

        # SignedConv uses pos/neg edge indices instead of edge_index
        if isinstance(self.conv, SignedConv):
            call.pop("edge_index", None)
            call.pop("edge_attr", None)
            call["pos_edge_index"] = edge_index
            call["neg_edge_index"] = edge_index

        # --- attention capture ---
        supports_attention = "return_attention_weights" in self._forward_params
        self.last_attention = None
        if self.capture_attention and supports_attention:
            out, attn = self.conv(**{**call, "return_attention_weights": True})
            if isinstance(attn, tuple) and len(attn) == 2:
                ei, aw = attn
                self.last_attention = (ei.detach().cpu(), aw.detach().cpu())
            else:
                self.last_attention = (None, torch.as_tensor(attn).detach().cpu())
        else:
            out = self.conv(**call)
            # Some convs (e.g. PANConv) return (node_features, auxiliary) — keep only the tensor
            if isinstance(out, tuple):
                out = out[0]

        if self.proj is not None:
            out = self.proj(out)
        return out

class myGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        hidden_channels: int,
        out_channels: int,
        conv_class=GATConv,
        conv_kwargs=None,
        heads=1,
        **kwargs
    ):
        super().__init__()
        self.conv_class = conv_class
        self.conv_kwargs = conv_kwargs or {}
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        try:
            ctor_params = set(signature(conv_class).parameters.keys())
        except Exception:
            ctor_params = set()

        use_heads = "heads" in ctor_params
        block_dim = self.hidden_channels * (heads if use_heads else 1)
        self.node_encoder = nn.Linear(in_channels, block_dim)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            conv = ConvAdapter(conv_class, in_dim=block_dim,
                               hidden_channels=self.hidden_channels,
                               heads=heads, base_kwargs=self.conv_kwargs)
            norm = nn.LayerNorm(block_dim)
            act = nn.ReLU(inplace=True)
            self.layers.append(DeepGCNLayer(conv=conv, norm=norm, act=act, block="res+"))

        self.norm_final = nn.LayerNorm(block_dim)
        self.fc = nn.Linear(block_dim, out_channels)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, return_attention=False, **kwargs):
        batch_vec = kwargs.get("batch", None)
        if x.dim() == 3:
            # Rare case: user still passes [N,B,F]
            N_per_graph, B, _ = x.shape
            x = rearrange(x, "n b c -> (b n) c")
            if batch_vec is None:
                batch_vec = torch.arange(B).repeat_interleave(N_per_graph).to(x.device)
        else:
            # Standard PyG: x is already [BN, C], so deduce info from batch vector
            if batch_vec is None:
                # FALLBACK: treat as one big graph (incompatible with attention aggregation)
                B = 1
                N_per_graph = x.shape[0]
                batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            else:
                B = int(batch_vec.max().item()) + 1
                N_per_graph = (batch_vec == 0).sum().item()

        x = self.node_encoder(x)
        x0 = x  # for DeepGCN residuals

        if return_attention:
            attentions = {
                "first_graph": [],  # list of (E_layer, heads)
                "mean": [],         # list of (E_layer, heads)
                "std": []           # list of (E_layer, heads)
            }

        for layer in self.layers:
            conv_adapter = layer.conv
            conv_adapter.capture_attention = bool(return_attention)

            out = layer(
                x, edge_index,
                edge_weight=edge_weight,
                x0=x0,
                batch=batch_vec
            )
            x = out

            if return_attention and conv_adapter.last_attention is not None:
                ei, aw = conv_adapter.last_attention
                # ei: [2, E_total], aw: [E_total, heads]
                E_total = aw.size(0)

                if B == 1:
                    # Single graph: trivial mapping
                    att_first = aw
                    att_mean = aw
                    att_std = torch.zeros_like(aw)
                else:
                    # Rebuild (B, E_graph, heads)
                    E_graph = E_total // B
                    att_reshaped = aw.view(B, E_graph, -1)  # [B, E_graph, H]

                    att_first = att_reshaped[0]             # [E_graph, H]
                    att_mean = att_reshaped.mean(dim=0)     # [E_graph, H]
                    att_std = att_reshaped.std(dim=0)       # [E_graph, H]

                attentions["first_graph"].append(att_first.detach().cpu())
                attentions["mean"].append(att_mean.detach().cpu())
                attentions["std"].append(att_std.detach().cpu())

        x = self.layers[0].act(self.norm_final(x))
        x = self.fc(x)
        return (x, attentions) if return_attention else x
    
class AdditiveGraphModel(nn.Module):
    r"""
    Additive Graph Model (GAM-like GNN).

    This model builds per-feature-group encoders and then applies one shared GNN
    backbone over a block-diagonal expansion of the original graph (one disjoint
    copy per feature group). The final per-node prediction is the average of
    group-wise outputs plus a learnable bias.

    Mathematical form
    -----------------
    :math:`h_g = \mathrm{Encoder}_g(x_g)`, :math:`y_g = \mathrm{GNN}(h_g, \mathcal{G})`,
    :math:`\hat{y} = b + \frac{1}{|G|} \sum_{g \in G} y_g`.

    Key points
    ----------
    - Single shared backbone over disjoint copies improves stability.
    - Avoids repeated backbone calls (prevents exploding activations).
    - Provides additive interpretability via ``group_outputs``.

    Parameters
    ----------
    feature_group_dims : dict[str, int]
        Mapping group name -> input feature dimension.
    num_layers : int
        Number of GNN layers in the shared backbone.
    hidden_channels : int
        Hidden dimension used by encoders and backbone.
    out_channels : int
        Output dimension per node.
    conv_class : type, optional
        PyG convolution class (default: ``GCNConv``).
    conv_kwargs : dict, optional
        Extra keyword arguments forwarded to the convolution constructor.

    Attributes
    ----------
    group_names : list[str]
        Ordered list of feature group names.
    num_groups : int
        Number of feature groups.
    encoders : nn.ModuleDict
        Per-group MLP encoders producing hidden representations.
    gnn_branch : myGNN
        Shared GNN backbone applied to concatenated group embeddings.
    bias : torch.nn.Parameter
        Learnable scalar bias added to predictions.
    group_index_map : dict[str, slice]
        Slices locating each group inside the concatenated feature tensor.
    """
    def __init__(
        self,
        feature_group_dims: dict,
        num_layers: int,
        hidden_channels: int,
        out_channels: int,
        conv_class=None,
        conv_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        if conv_class is None:
            conv_class = GCNConv
        conv_kwargs = conv_kwargs or {}

        self.feature_group_dims = feature_group_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.heads = conv_kwargs.get("heads", 1)
        self.group_names = list(feature_group_dims.keys())
        self.num_groups = len(self.group_names)

        self.encoders = nn.ModuleDict()
        for gname, dim in feature_group_dims.items():
            self.encoders[gname] = nn.Sequential(
                nn.Linear(dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )

        self.gnn_branch = myGNN(
            in_channels=hidden_channels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            conv_class=conv_class,
            conv_kwargs=conv_kwargs,
        )

        self.bias = nn.Parameter(torch.zeros(1))

        self.group_index_map = {}
        offset = 0
        for gname, dim in feature_group_dims.items():
            self.group_index_map[gname] = slice(offset, offset + dim)
            offset += dim

    @staticmethod
    def _repeat_edge_index(edge_index: torch.Tensor, num_nodes: int, repeats: int, device=None):
        """
        Create a block-diagonal edge_index with ``repeats`` disjoint copies of the graph.

        Each copy is shifted by ``k * num_nodes`` in node indices.

        :param edge_index: Base edge index [2, E].
        :type edge_index: torch.Tensor
        :param num_nodes: Number of nodes in the original graph.
        :type num_nodes: int
        :param repeats: Number of disjoint copies.
        :type repeats: int
        :param device: Optional device for the output tensor.
        :type device: torch.device | str | None
        :returns: Expanded edge index [2, repeats * E].
        :rtype: torch.Tensor
        """
        if edge_index is None:
            return None

        edge_index = edge_index.to(device)
        row, col = edge_index
        all_rows = []
        all_cols = []

        for k in range(repeats):
            offset = k * num_nodes
            all_rows.append(row + offset)
            all_cols.append(col + offset)

        row_big = torch.cat(all_rows, dim=0)
        col_big = torch.cat(all_cols, dim=0)
        return torch.stack([row_big, col_big], dim=0)

    @staticmethod
    def _repeat_edge_weight(edge_weight: torch.Tensor, repeats: int, device=None):
        """
        Repeat edge weights for block-diagonal expansion.

        :param edge_weight: Edge weights [E].
        :type edge_weight: torch.Tensor
        :param repeats: Number of repetitions.
        :type repeats: int
        :param device: Optional device.
        :type device: torch.device | str | None
        :returns: Concatenated edge weights [repeats * E] or None.
        :rtype: torch.Tensor | None
        """
        if edge_weight is None:
            return None
        edge_weight = edge_weight.to(device)
        return edge_weight.repeat(repeats)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        mask=None,  # kept for API compatibility, unused
        return_attention: bool = False,
        return_group_outputs: bool = False,
        **kwargs
    ):
        """
        Forward pass.

        Fast path (``return_attention=False``) constructs a block-diagonal graph and
        performs one shared GNN pass. Slow path (``return_attention=True``) iterates
        per group to collect attention maps.

        :param x: Node features concatenated by group, shape [N, F_total].
        :type x: torch.Tensor
        :param edge_index: Graph connectivity [2, E].
        :type edge_index: torch.Tensor
        :param edge_weight: Optional edge weights [E].
        :type edge_weight: torch.Tensor | None
        :param mask: Unused placeholder (trainer compatibility).
        :type mask: Any | None
        :param return_attention: If True, returns per-group attention statistics.
        :type return_attention: bool
        :param return_group_outputs: If True, also return per-group node outputs.
        :type return_group_outputs: bool
        :returns:
            - y_hat: Tensor [N, out_channels] (always)
            - group_outputs (dict[str, Tensor]) if ``return_group_outputs`` is True
            - attention_per_group (dict[str, dict]) if ``return_attention`` is True
        :rtype: torch.Tensor | tuple
        """
        device = self.bias.device

        x = x.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        N = x.size(0)
        G = self.num_groups

        # Fast, stable path: no attention requested
        if not return_attention:
            h_list = []
            for gname in self.group_names:
                sl = self.group_index_map[gname]
                x_g = x[:, sl]  # [N, d_g]
                if x_g.dim() == 1:
                    x_g = x_g.unsqueeze(-1)
                h_g = self.encoders[gname](x_g)  # [N, hidden]
                h_list.append(h_g)

            H = torch.stack(h_list, dim=0)           # [G, N, hidden]
            H_flat = H.reshape(G * N, self.hidden_channels)
            edge_index_big = self._repeat_edge_index(edge_index, num_nodes=N, repeats=G, device=device)
            edge_weight_big = self._repeat_edge_weight(edge_weight, repeats=G, device=device)

            y_flat = self.gnn_branch(
                H_flat,
                edge_index_big,
                edge_weight=edge_weight_big,
                return_attention=False,
            )  # [G*N, out_channels]

            if y_flat.dim() == 1:
                y_flat = y_flat.unsqueeze(-1)
            Y = y_flat.view(G, N, self.out_channels)

            group_outputs = {
                gname: Y[i] for i, gname in enumerate(self.group_names)
            }  # gname -> [N, out_channels]

            total = Y.sum(dim=0) / float(G)  # [N, out_channels]
            y_hat = total + self.bias        # [N, out_channels] (broadcast)

            if not return_group_outputs:
                return y_hat
            else:
                return y_hat, group_outputs

        # Slow, per-group path: needed if return_attention == True.
        total = torch.zeros(N, self.out_channels, device=device)
        group_outputs = {}
        attention_per_group = {}

        for gname, sl in self.group_index_map.items():
            x_g = x[:, sl]  # [N, d_g]
            if x_g.dim() == 1:
                x_g = x_g.unsqueeze(-1)
            h_g = self.encoders[gname](x_g)  # [N, hidden]

            y_g, attn_g = self.gnn_branch(
                h_g,
                edge_index,
                edge_weight=edge_weight,
                return_attention=True,
            )

            if y_g.dim() == 1:
                y_g = y_g.unsqueeze(-1)

            group_outputs[gname] = y_g
            attention_per_group[gname] = attn_g
            total = total + y_g
            
        y_hat = total + self.bias

        if not return_group_outputs and not return_attention:
            return y_hat
        if not return_group_outputs and return_attention:
            return y_hat, attention_per_group
        if return_attention and not return_group_outputs:
            return y_hat, attention_per_group
        if return_attention and return_group_outputs:
            return y_hat, group_outputs, attention_per_group
            
class GCNEncoder(torch.nn.Module):
    """
    Simple 2-layer Graph Convolutional Network encoder.

    This encoder maps node features into a compact latent space using
    two GCNConv layers and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    out_channels : int
        Dimension of the latent embedding space.

    Examples
    --------
    >>> enc = GCNEncoder(in_channels=32, out_channels=16)
    >>> z = enc(x, edge_index)
    >>> z.shape
    torch.Size([N, 16])
    """
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels) 
        self.conv2 = GCNConv(2 * out_channels, out_channels) 

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, F]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.

        Returns
        -------
        torch.Tensor
            Latent node representations of shape ``[N, out_channels]``.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGNNEncoder(torch.nn.Module):
    """
    Variational Graph Encoder producing mean and log-variance embeddings
    for Variational Graph Autoencoders (VGAE) or graph-based latent models.

    Supports both GCN and GraphSAGE convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    out_channels : int
        Latent embedding dimension.
    conv : {'gcn', 'sage'}, default='gcn'
        Type of graph convolution to use.

    Attributes
    ----------
    conv1 : torch_geometric.nn.MessagePassing
        First convolution layer (shared for both mu/logstd branches).
    conv_mu : torch_geometric.nn.MessagePassing
        Convolution layer producing mean embeddings.
    conv_logstd : torch_geometric.nn.MessagePassing
        Convolution layer producing log-variance embeddings.

    Examples
    --------
    >>> enc = VariationalGNNEncoder(in_channels=32, out_channels=16, conv='sage')
    >>> mu, logstd = enc(x, edge_index)
    >>> mu.shape, logstd.shape
    (torch.Size([N, 16]), torch.Size([N, 16]))
    """
    def __init__(self, in_channels, out_channels, conv='gcn'):
        super(VariationalGNNEncoder, self).__init__()
        if conv == 'gcn':
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        else:
            self.conv1 = SAGEConv(in_channels, 2 * out_channels)
            self.conv_mu = SAGEConv(2 * out_channels, out_channels)
            self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Compute latent mean and log-variance representations.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[N, F]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.

        Returns
        -------
        tuple of torch.Tensor
            Mean and log-variance tensors, each of shape ``[N, out_channels]``.
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)