import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from block import *


class BMT(nn.Module): # BIT
    def __init__(self, dim=512, num_heads=8, memory_slots=30, depth=3, cls_num=3, Fusion=True, lamd=False, index=0,
                 return_lamd=False, fusion_func='cat', cls_emb=True):
        super().__init__()
        self.memory_initers = nn.ModuleList([MemoryInit(memory_slots, dim)
            for _ in range(8)])
        ############  gaters are PIN.
        self.forward_gaters = nn.ModuleList([Adaptive_Memory_Gate_Fusion(dim, num_heads)
                                            for _ in range(4)]) # memory-gate
        self.forward_gaters_depth = nn.ModuleList([self.forward_gaters
                                                   for _ in range(depth)])
        #############
        self.backward_gaters = nn.ModuleList([Adaptive_Memory_Gate_Fusion(dim, num_heads)
                                             for _ in range(4)])  # memory-gat
        self.backward_gaters_depth = nn.ModuleList([self.backward_gaters
                                                    for _ in range(depth)])
        ########### blocks are BIE
        self.forward_blocks = nn.ModuleList([Memory_augmented_Interactive_Block(dim, num_heads)
            for _ in range(depth)])
        self.backward_blocks = nn.ModuleList([Memory_augmented_Interactive_Block(dim, num_heads)
            for _ in range(depth)])
        ########
        self.index = index  #0 or -1
        self.cls_emb = cls_emb
        if cls_emb:
            self.cls_b_i = nn.Parameter(torch.zeros(1, 1, dim))  # cat
            self.cls_b_t = nn.Parameter(torch.zeros(1, 1, dim))  # cat
        self.Fusion = Fusion
        self.return_lamd = return_lamd
        self.fusion_func = fusion_func
        if Fusion:
            #self.head = Bi_fusion(dim=dim, down_dim=dim, num_cls=cls_num, lamd=lamd)
            #self.head = Ada_Bi_fusion(dim=dim, down_dim=dim, num_cls=cls_num, lamd=lamd)
            self.head = Ada_Bi_fusion_v2(dim=dim, down_dim=dim, num_cls=cls_num, lamd=return_lamd, fusion_func=self.fusion_func)

        else:
            self.head = nn.Sequential(
                nn.Linear(4*dim, 2*dim),
                nn.Linear(2*dim, cls_num)
            )

        self.apply(self._init_weights)
    # 对LN 和 Linear 做初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image, text):
        forward_image = image
        backward_image = image.__reversed__()
         #
        forward_text = text
        backward_text = text.__reversed__()
        # 这里的初始化可以进行更改，后续进行尝试，看看初始化image时用text的效果怎么样
        fw_i_mk, fw_i_mv = self.memory_initers[0](forward_image), self.memory_initers[1](forward_text)
        fw_t_mk, fw_t_mv = self.memory_initers[2](forward_text), self.memory_initers[3](forward_image)
        #
        bw_i_mk, bw_i_mv = self.memory_initers[4](backward_image), self.memory_initers[5](backward_text)
        bw_t_mk, bw_t_mv = self.memory_initers[6](backward_text), self.memory_initers[7](backward_image)

        # fw_i_mk, fw_i_mv = self.memory_initers[0](forward_image), self.memory_initers[1](forward_image)
        # fw_t_mk, fw_t_mv = self.memory_initers[2](forward_text), self.memory_initers[3](forward_text)
        # #
        # bw_i_mk, bw_i_mv = self.memory_initers[4](backward_image), self.memory_initers[5](backward_image)
        # bw_t_mk, bw_t_mv = self.memory_initers[6](backward_text), self.memory_initers[7](backward_text)

        B = image.shape[0]
        # i_N = image.shape[1]
        # t_N = text.shape[1]
        #
        # backward_image = torch.cat([self.cls_b_i.expand(B, 1, -1), backward_image[:, 0:i_N-1, :]], dim=1)
        # backward_text = torch.cat([self.cls_b_t.expand(B, 1, -1), backward_text[:, 0:t_N-1, :]], dim=1)
        if self.cls_emb:
            backward_image = torch.cat([self.cls_b_i.expand(B, 1, -1), backward_image], dim=1)
            backward_text = torch.cat([self.cls_b_t.expand(B, 1, -1), backward_text], dim=1)

        for fw_gaters, bw_gaters, fw_block, bw_block in zip(self.forward_gaters_depth, self.backward_gaters_depth,
                                                            self.forward_blocks, self.backward_blocks):


            # fw
            fw_i_mk_, fw_i_mv_ = fw_gaters[0](fw_i_mk, bw_i_mk), fw_gaters[1](fw_i_mv, bw_i_mv)
            fw_t_mk_, fw_t_mv_ = fw_gaters[2](fw_t_mk, bw_t_mk), fw_gaters[3](fw_t_mv, bw_t_mv)
            # bw
            bw_i_mk_, bw_i_mv_ = bw_gaters[0](bw_i_mk, fw_i_mk), bw_gaters[1](bw_i_mv, fw_i_mv)
            bw_t_mk_, bw_t_mv_ = bw_gaters[2](bw_t_mk, fw_t_mk), bw_gaters[3](bw_t_mv, fw_t_mv)
            # # fw
            # fw_i_mk_, fw_i_mv_ = fw_gaters[0](fw_i_mk, bw_i_mk, forward_image), fw_gaters[1](fw_i_mv, bw_i_mv, forward_text)
            # fw_t_mk_, fw_t_mv_ = fw_gaters[2](fw_t_mk, bw_t_mk, forward_text), fw_gaters[3](fw_t_mv, bw_t_mv, forward_image)
            # # bw
            # bw_i_mk_, bw_i_mv_ = bw_gaters[0](bw_i_mk, fw_i_mk, backward_image), bw_gaters[1](bw_i_mv, fw_i_mv, backward_text)
            # bw_t_mk_, bw_t_mv_ = bw_gaters[2](bw_t_mk, fw_t_mk, backward_text), bw_gaters[3](bw_t_mv, fw_t_mv, backward_image)

            # fw
            forward_image, forward_text = fw_block(forward_image, forward_text,
                                      fw_i_mk_, fw_i_mv_,
                                      fw_t_mk_, fw_t_mv_)

            backward_image, backward_text = bw_block(backward_image, backward_text,
                                      bw_i_mk_, bw_i_mv_,
                                      bw_t_mk_, bw_t_mv_)

            #
            # forward_image, forward_text = fw_i_o, fw_t_o
            # backward_image, backward_text = bw_i_o, bw_t_o
            #
            fw_i_mk, fw_i_mv = fw_i_mk_, fw_i_mv_
            fw_t_mk, fw_t_mv = fw_t_mk_, fw_t_mv_
            bw_i_mk, bw_i_mv = bw_i_mk_, bw_i_mv_
            bw_t_mk, bw_t_mv = bw_t_mk_, bw_t_mv_
        # forward_o = torch.cat([forward_image[:, 0, :], forward_text[:, 0, :]], dim=-1)
        # backward_o = torch.cat([backward_image[:, 0, :], backward_text[:, 0, :]], dim=-1)
        # o = torch.cat([forward_o, backward_o], dim=-1)
        # index = self.index = 0 or -1
        if self.Fusion:
            o = self.head(forward_image[:, 0, :], forward_text[:, 0, :],
                          backward_image[:, self.index, :], backward_text[:, self.index, :])
            if self.return_lamd:
                return o[0], o[1]
            else:
                return o
        else:  # -1 or 0 看看效果。
            o = torch.cat([forward_image[:, 0, :], forward_text[:, 0, :],
                           backward_image[:, self.index, :], backward_text[:, self.index, :]], dim=-1)
            o = self.head(o)
            return o

if __name__ == '__main__':
    model = BMT(depth=6, return_lamd=True).cuda()
    image = torch.randn(16, 197, 512).cuda()
    text = torch.randn(16, 77, 512).cuda()
    o = model(image, text)


