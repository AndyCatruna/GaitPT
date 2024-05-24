import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
	def __init__(self, *, dim, dim_out, patch_size):
		super().__init__()
		self.dim = dim
		self.dim_out = dim_out
		self.patch_size = patch_size
		self.proj = nn.Linear(patch_size * dim, dim_out)

	def forward(self, fmap):
		p = self.patch_size
		fmap = rearrange(fmap, 'b len (joint patch) c -> b len joint (c patch)', patch=p)

		fmap = self.proj(fmap)
		return fmap

class SpatialAttention(nn.Module):
	def __init__(self, args, d_model, num_patches, depth):
		super().__init__()

		self.to_embedding = Rearrange('b len joint c -> (b len) joint c')

		self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

		spatial_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=args.nheads, dim_feedforward=d_model * args.mlp_amplifier, dropout=args.dropout, batch_first=True, norm_first=True, activation='gelu') 
		self.spatial_transformer = TransformerEncoder(spatial_encoder_layer, num_layers=depth)

		self.rearrange_back = Rearrange('(b len) joint c -> b len joint c', len=args.sequence_length)

		self.norm = nn.LayerNorm(normalized_shape=d_model)

		self.use_residual = args.residual
		self.residual = lambda x: x

	def forward(self, x):
		x = self.to_embedding(x)

		if self.use_residual:
			res = self.residual(x)

		x += self.spatial_pos_embed
		x = self.spatial_transformer(x)
		x = self.norm(x)

		if self.use_residual:
			x += res

		x = self.rearrange_back(x)

		return x

class TemporalAttention(nn.Module):
	def __init__(self, args, d_model, stage_depth, num_patches):
		super().__init__()

		self.to_embedding = Rearrange('b len joint c -> (b joint) len c')

		length = args.sequence_length + 1
		self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
		self.rearrange_result = Rearrange('(b joint) len c -> b len (joint c)', len=length, joint=num_patches)
		
		self.temporal_pos_embed = nn.Parameter(torch.zeros(1, length, d_model))

		encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=args.nheads, dim_feedforward=d_model * args.mlp_amplifier, dropout=args.dropout, batch_first=True, norm_first=True, activation='gelu') 
		self.transformer = TransformerEncoder(encoder_layer, num_layers=stage_depth)

		self.rearrange_back = Rearrange('(b joint) len c -> b len joint c', len=args.sequence_length, joint=num_patches)

		self.norm = nn.LayerNorm(d_model)

		self.use_residual = args.residual
		self.residual = lambda x: x

	def forward(self, x):
		x = self.to_embedding(x)

		if self.use_residual:
			res = self.residual(x)

		b, _, _ = x.shape
		cls_tokens = repeat(self.temporal_cls_token, '1 1 d -> b 1 d', b = b)
		x = torch.cat([cls_tokens, x], dim=1)

		x += self.temporal_pos_embed
		x = self.transformer(x)
		x = self.norm(x)
		
		ret = self.rearrange_result(x)[:, 0]
		x = x[:, 1:]
		
		if self.use_residual:
			x += res

		x = self.rearrange_back(x)
		
		return x, ret

class StageModule(nn.Module):
	def __init__(self, args, stage_number):
		super().__init__()

		self.patch_embedding = PatchEmbedding(dim=args.input_size[stage_number], dim_out=args.dim_size[stage_number], patch_size=args.patch_size[stage_number])
		self.spatial_module = SpatialAttention(args, d_model=args.dim_size[stage_number], num_patches=args.num_patches[stage_number], depth=args.stage_depths[stage_number])
		self.temporal_module = TemporalAttention(args, d_model=args.dim_size[stage_number],  num_patches=args.num_patches[stage_number], stage_depth=args.stage_depths[stage_number])
		
		if stage_number == 3:
			self.spatial_module = nn.Identity()

		self.aggregate = args.aggregate
		self.stage_number = stage_number

		self.use_residual = args.residual
		self.residual = lambda x: x

	def forward(self, x):
		x = self.patch_embedding(x)

		x = self.spatial_module(x)

		x, ret = self.temporal_module(x)

		if self.aggregate or self.stage_number == 3:
			return x, ret

		return x

class GaitPT(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.args = args
		self.layers = nn.ModuleList([])

		output_size = 0
		for i, depth in enumerate(args.stage_depths):
			if depth > 0:
				self.layers.append(StageModule(args, stage_number=i))
				output_size += args.output_size[i]
			else:
				self.layers.append(PatchEmbedding(dim=args.input_size[i], dim_out=args.dim_size[i], patch_size=args.patch_size[i]))

		self.rearrange = Rearrange('b period emb -> b emb period')

		self.projection = nn.Linear(
			in_features = output_size,
			out_features = args.projection_size,
		)

		self.aggregate = args.aggregate

	def forward(self, x):
		all_ret = []
		for i, layer in enumerate(self.layers):
			if i == 3 or (self.aggregate and self.args.stage_depths[i] > 0):
				x, ret = layer(x)
				all_ret += [ret]
			else:
				x = layer(x)
			
		if self.aggregate:
			x = torch.cat(all_ret, dim=1)
		else:
			x = ret

		projection = self.projection(x)
		projection = F.normalize(projection)
		
		return projection
