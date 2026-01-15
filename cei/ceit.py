import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2  # 切片数量(2048//32)**2==64**2==4096
        patch_dim = channels * patch_size ** 2  # 一张2048x2048的图被分为32x32大小的4096块,每一块3通道,将每一块展平:32x32x3=3072 所以patch_dim维度为:3072

        self.patch_size = patch_size  # patch_size:16

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码:[1,4096+1,dim=512]
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # 将3072维度(像素点)embeding到512维度的空间
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 每一个维度都有一个类别的标志位
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位符

        self.mlp_head = nn.Sequential(  # 分类头
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size  # 32
        '''
        #img:[batch, 3, 2048, 2048]
        #'batch 3 (h 32) (w 32)'->'batch (h,w) (32 32 3)' 
        将图像分块,且每块展平(像素为单位连接起来)
        '''
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p,
                      p2=p)  # [batch, 4096, 3072]　4096块,每一块展开为3072维向量
        x = self.patch_to_embedding(x)  # [batch, 4096, 512] 将3072维度的像素嵌入到512的空间
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [1,1,512]->[b,1,512]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch,4096+1,512]
        x += self.pos_embedding[:, :(n + 1)]  # 加上位置编码信息
        '''
        以上步骤干的事情:
        - 输入图片分块->展平:[batch,c,h,w]->[batch,num_patch,c*patch_size*patch_size]
        - 原始的像素嵌入到指定维度(dim):[batch,num_patch,c*patch_size*patch_size]->[batch,num_patch,dim]
        - 每一个样本的每一个维度都加入类别token,给分片的图像多加一片,专门用来表示类别
            - [batch,num_patch,dim]->[batch,num_patch+1,dim]
        - 给所有的"片(patch)"加入位置编码信息.这里的位置编码初始化为随机数,是通过网络学习出来的
        以上步骤产生的输出结果即可送入到Transformer里面进行编码
        [batch,num_patch+1,dim]经过transformer的编码将会出来一个[batch,num_patch+1,dim]的向量
        '''
        x = self.transformer(x)  # [batch,num_patch+1,dim]->[batch,num_patch+1,dim]

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # [batch,dim]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = x.reshape(20,1,96,96)
        return x