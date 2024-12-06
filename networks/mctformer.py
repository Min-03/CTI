import torch
import torch.nn as nn
from functools import partial
from networks.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from networks.vision_transformer import Block, PatchEmbed
from torch.cuda.amp import autocast

import math

import pdb


__all__ = [
           'deit_small_MCTformerV2_CTI',
           ]

class MCTformerV2_CTI(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes_with_fg = self.num_classes + 1             #FG
       
        self.head = nn.Conv2d(self.embed_dim, self.num_classes_with_fg, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        num_patches = self.patch_embed.num_patches

        self.bg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes_with_fg, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        self.tokenizer =  nn.Sequential(nn.Linear(384, 384*5),
                                        nn.ReLU(),
                                        nn.Linear(384*5, 384*self.num_classes_with_fg))
       
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed
    
    def forward_features(self, x, swap_ctk=None, swap_idx=6, noise_weight=-1.0, fuse_factor=0.5):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        #######Tokenizing with background token######
        cls_fg = self.cls_token.expand(B, -1,-1)
        cls_bg = self.bg_token.expand(B, -1,-1)
        cls_tokens = self.tokenizer(cls_fg.view(B,-1)).view(B,self.num_classes_with_fg,self.embed_dim)

        cls_tokens[:,1:,:]+=cls_fg
        cls_tokens[:,0,:] +=cls_bg.squeeze(1)

        cls_tokens = cls_tokens + self.pos_embed_cls
        #############################################

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        attn_weights = []
        ctk_list = []
        query_list = []

        if not (noise_weight < 0): #temporary best value
            swap_idx = 3

        for i, blk in enumerate(self.blocks):
            if (swap_ctk is not None or not (noise_weight < 0)) and i==(swap_idx + 1): #BE careful for the SWAP IDX
                if not (noise_weight < 0):
                    ctk = x[:, :self.num_classes_with_fg]
                    swap_ctk = ctk.clone() + noise_weight * torch.randn_like(ctk.detach()) * ctk.std().detach()
                if swap_ctk.size(1) == self.num_classes_with_fg:
                    x[:, :self.num_classes_with_fg] =  fuse_factor * x[:, :self.num_classes_with_fg] + (1 - fuse_factor) * swap_ctk    # FG # swap token with fg token
                elif swap_ctk.size(1) == self.num_classes:
                    x[:, 1:self.num_classes_with_fg] =  fuse_factor * x[:, 1:self.num_classes_with_fg] + (1 - fuse_factor) * swap_ctk    # FG # swap token with fg token
                x, weights_i,qkv, _ = blk(x)
                attn_weights.append(weights_i)
                ctk_list.append(x[:, 0:self.num_classes_with_fg])
                query_list.append(qkv[0][:, 0:self.num_classes_with_fg])
                
            else:
                x, weights_i,qkv, _ = blk(x)
                attn_weights.append(weights_i)
                ctk_list.append(x[:, 0:self.num_classes_with_fg])
                query_list.append(qkv[0][:, 0:self.num_classes_with_fg])

        return x[:, 0:self.num_classes_with_fg], x[:, self.num_classes_with_fg:], attn_weights , ctk_list, query_list

    def forward(self, x, ctk=None, swap_idx=None, return_att=False, n_layers=6, noise_weight=-1.0, fuse_factor=0.5):  # FG input ctk doesn't include fg token
        w, h = x.shape[2:]
        if ctk == None:
            x_cls, x_patch, attn_weights, ctk_list, query_list = self.forward_features(x, noise_weight=noise_weight, fuse_factor=fuse_factor)
        else: 
            x_cls, x_patch, attn_weights, ctk_list, query_list = self.forward_features(x,ctk,swap_idx, fuse_factor=fuse_factor)

        n, p, c = x_patch.shape

        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
            #########################
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
            #########################
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        feat = x_patch
        
        x_patch = self.head(x_patch)

        x_patch_logits = self.avgpool(x_patch[:,1:,:,:]).squeeze(3).squeeze(2) ######ORIGINAL

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch  # B * C * 14 * 14
        
        ##########################code for feature noise######################################
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes_with_fg, self.num_classes_with_fg:].reshape([n, c, h, w])
        
        if ctk is not None or swap_idx is not None:
            mtatt_as = attn_weights[swap_idx+1:].sum(0)[:, 0:self.num_classes_with_fg, self.num_classes_with_fg:].reshape([n, c, h, w])

        cams = mtatt * feature_map  # B * C * 14 * 14

        patch_attn = attn_weights[:, :, self.num_classes_with_fg:, self.num_classes_with_fg:]

        if swap_idx is not None:
            patch_attn = torch.sum(patch_attn[swap_idx+1:], dim=0) #B 196 196
        else:
            patch_attn = torch.sum(patch_attn, dim=0) #B 196 196
        ##########################################################################
        x_cls_logits = x_cls[:, 1:].mean(-1)    # FG without fg
        ##########################################################################

        rcams = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],cams.shape[1], -1, 1)).reshape(cams.shape[0],cams.shape[1], h, w) #(B 1 N2 N2) * (B,20,N2,1)

        outs = {}
        outs['cls']= x_cls_logits
        outs['pcls']= x_patch_logits
        outs['cams']= F.relu(x_patch)
        outs['Sattn']= attn_weights
        outs['fcams']= F.relu(x_patch) * mtatt
        outs['attn']= patch_attn
        outs['mtatt']= mtatt
        if ctk is not None or swap_idx is not None:
            outs['mtatt_as']= mtatt_as
            outs['fcams_as']= F.relu(x_patch) * mtatt_as
        outs['rcams']= rcams
        outs['ctk']= ctk_list
        outs['query'] = query_list

        # outs['logit_fg'] = x_fg_logit

        if return_att:
            return rcams  
        else:
            return outs


@register_model
def deit_small_MCTformerV2_CTI(pretrained=False, **kwargs):
    model = MCTformerV2_CTI(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model