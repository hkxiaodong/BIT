import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    '''
    self-attention
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # b, 8, N, dim/8
        # print(q.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention_without_pro(nn.Module):
    '''
    cross-attention
    inputs: X and Y from image and text. X represent local input, Y represent other input.
            shapes of X and Y may be different. X as q, Y as k,v. X interact with Y to query sentimental information
            embedded in Y.
    output:
    '''
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x1, x2):
        B, N_1, C = x1.shape
        N_2 = x2.shape[1]  # n2 may not = n1
        # print(x1.device)
        # print(x2.device)
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       # b , h , N , dim/h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C) # q's shape.

        return x

class Cross_Attention_with_pro(nn.Module):
    '''
    cross-attention
    inputs: X and Y from image and text. X represent local input, Y represent other input.
            shapes of X and Y may be different. X as q, Y as k,v. X interact with Y to query sentimental information
            embedded in Y.
    output:
    '''
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.project = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N_1, C = x1.shape
        N_2 = x2.shape[1]  # n2 may not = n1
        # print(x1.device)
        # print(x2.device)
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       # b , h , N , dim/h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C) # q's shape.
        x = self.project(x)
        return x

class MemoryInit(nn.Module): # prefix init
    def __init__(self, n_memory_cells, dim):
        super(MemoryInit, self).__init__()
        # init memory
        self.n_memory_cells = n_memory_cells
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )

    def forward(self, input_states):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            :returns  (N, M, D)
        """
        pooled_input_states = torch.sum(input_states, dim=1)   # (N, D)
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory


class Memory_attention(nn.Module):
    '''
    cross-attention with updated memory matrix m_k and m_v
    inputs: X and Y from image and text. X represents local input, Y represents other input.
            shapes of X and Y may be different. X as q, Y as k,v. X interact with Y to query the sentimental
            information embedded in Y.
    output:
    '''
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        #self.scale = dim ** -0.5
        self.scale = self.head_dim ** -0.5
        #
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x1, x2, memory_k, memory_v, perfix=True):
        B, N_1, C = x1.shape
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        N_2 = x2.shape[1]  # n2 may not = n1
        # b,n2+img_d,c2
        if perfix:
            k = self.k(x2)
            v = self.v(x2)
            memory_solts = memory_k.shape[1] # the length of memory slots
            k = torch.cat([k, memory_k], dim=1).reshape(B,  N_2+memory_solts, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = torch.cat([v, memory_v], dim=1).reshape(B,  N_2+memory_solts, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # b,8,n2+m,64
        else:
            k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #
            v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # b,h,n1,64 @ b,h,64,n2+m-->b,h,n1,n2+m
        #b, h, n1, n2 + m @ b,h,n2+m,64-->b,h,n1,64-->b,n1,h,64-->b,n1,h*64=c1
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Adaptive_Memory_Gate_Fusion(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = Cross_Attention_without_pro(dim=dim, num_heads=num_heads)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        q_m = self.cross_attn(x1, x2)
        gate = torch.sigmoid(q_m + self.linear1(x1))
        out = gate * self.linear2(x1)
        '''
        可以输出gate 来查看是前向记忆重要还是后向记忆重要
        '''
        return out

class Cross_Memory_Block(nn.Module): # PIA
    '''
    PIA
    '''
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = Memory_attention(dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 3)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, memory_k, memory_v, perfix=True):
        out = self.norm1(x1 + self.cross_attn(x1, x2, memory_k, memory_v, perfix=perfix))
        out = out + self.norm2(self.mlp(out))
        return out



class Memory_augmented_Interactive_Block(nn.Module): # HMPIA
    '''
    HMPIA
    '''
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.image_block = Cross_Memory_Block(dim, num_heads=num_heads)
        self.text_block = Cross_Memory_Block(dim, num_heads=num_heads)

    def forward(self, x1, x2, mk_i, mv_i, mk_t, mv_t, perfix=True):
        image_out = self.image_block(x1, x2, mk_i, mv_i, perfix=perfix)
        text_out = self.text_block(x2, x1, mk_t, mv_t, perfix=perfix)
        return image_out, text_out


class Ada_Bi_fusion_v2(nn.Module): # TAGF
    def __init__(self, dim, down_dim, num_cls, lamd=False, fusion_func='cat'):
        super().__init__()
        self.f_dowm_linear = nn.Linear(dim*2, down_dim)
        self.b_dowm_linear = nn.Linear(dim*2, down_dim)

        self.linear_f = nn.Linear(2 * dim + down_dim, dim)
        self.linear_b = nn.Linear(2 * dim + down_dim, dim)
        self.fusion_func = fusion_func

        if self.fusion_func == 'sum':
            self.linear_1 = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        if self.fusion_func == 'cat':
            self.linear_1 = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        self.linear_o = nn.Sequential(
            nn.Linear(dim, num_cls)
        )
        self.return_lamd = lamd

    def forward(self, Fi, Ft, Bi, Bt):
        cat_f = torch.cat([Fi, Ft], dim=-1)
        Fo = torch.cat([cat_f, self.f_dowm_linear(cat_f)], dim=-1)
        Fo = self.linear_f(Fo)

        cat_b = torch.cat([Bi, Bt], dim=-1)
        Bo = torch.cat([cat_b, self.f_dowm_linear(cat_b)], dim=-1)
        Bo = self.linear_b(Bo)


        if self.fusion_func == 'cat':
            cat_all = torch.cat([Fo, Bo], dim=-1)
            gate1 = self.linear_1(cat_all)

        if self.fusion_func == 'sum':
            gate1 = self.linear_1(Fo+Bo)

        o = self.linear_o(gate1*Fo + (1-gate1)*Bo)

        if self.return_lamd:
            return o, gate1
        else:
            return o


if __name__ == '__main__':
    x1 = torch.randn(16, 77, 512)
    x2 = torch.randn(16, 197, 512)
    memory = torch.randn(16, 50, 512)

