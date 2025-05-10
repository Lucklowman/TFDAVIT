import torch
from torch import nn
import math
def window_partition(x, window_size: int):
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size,C)
    return windows


def window_reverse(windows, window_size: int, N: int):
    num_windows = N // window_size
    B = int(windows.shape[0] / num_windows)
    x = windows.view(B, num_windows, window_size,-1)
    x = x.permute(0,2,1,3)
    x = x.contiguous().view(B, N, -1)
    return x

class WindowAttentiondwconve2(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.gwproj = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(1)) 
        self.b = nn.Parameter(torch.zeros(1)) 
        self.qkv2 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
    def forward(self, x,x2,x3):
        B_, N, C = x.shape
        B1,N1,C1 = x2.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qkv1  = self.qkv2(x3).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        resv = v1
        gw = self.gwproj(x2)
        gw = gw.reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        v = v1
        q = q1 * self.scale * gw
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x = (attn @ v)
        x = x + resv * self.a 
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class ConvPosEnc1D(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc1D, self).__init__()
        self.proj = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim
        )

    def forward(self, x):
        B, N, C = x.shape
        feat = x.transpose(1, 2)
        feat = self.proj(feat)
        feat = feat.transpose(1, 2)
        x = x + feat
        return x

class ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0.1,ffn=True):
        super().__init__()
        self.cpe = nn.ModuleList([ConvPosEnc1D(dim=dim, k=3),
                                  ConvPosEnc1D(dim=dim, k=3)])
        self.ffn = ffn
        self.dim = dim
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ECSAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        if self.ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.drop_path = nn.Dropout(drop_rate)
        self.fan = FeedForwardFAN(inputdim=dim,outputdim=dim,rate=0.25)


    def forward(self, x):
        shortcut = self.cpe[0](x)
        x = self.norm1(shortcut)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = self.cpe[1](x)
        x = x + self.drop_path(self.fan(self.norm2(x)))
        return x

class ECSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.k = int(torch.round((torch.log2(torch.tensor(dim, dtype=torch.float32)) / 2 + 1 / 2)).item())
        if self.k % 2 == 0:
            self.k += 1
        self.conv = nn.Conv1d(in_channels=head_dim,out_channels=head_dim,kernel_size=self.k,stride=1,padding="same")
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=head_dim,out_features=head_dim)
        self.dwconv1 = nn.Conv1d(in_channels=head_dim,out_channels=head_dim,kernel_size=3,stride=1,padding="same")
        self.dwconv2 = nn.Conv1d(in_channels=head_dim,out_channels=head_dim,kernel_size=5,stride=1,padding="same")  
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        res = v
        res = res.reshape(-1,N, C // self.num_heads)
        res = res.permute(0,2,1)
        res = self.avg(res)
        res = self.conv(res)
        res = self.sigmoid(res)
        res = res.permute(0,2,1)
        res = res.reshape(B,self.num_heads,1, C // self.num_heads)

        v = v.reshape(-1,N, C // self.num_heads).permute(0,2,1)
        v1 = self.dwconv1(v)
        v2 = self.dwconv2(v)
        v = v1 + v2
        v = v.permute(0,2,1).reshape(B,self.num_heads,N, C // self.num_heads)

        k = k * self.scale
        attention1 = k.transpose(-1, -2) @ q
        attres = attention1

        attention1 = self.relu(attention1)
        attention1 = attention1.reshape(-1,C // self.num_heads)
        attention1 = self.fc(attention1)
        attention1 = attention1.reshape(B,self.num_heads,C // self.num_heads,C // self.num_heads)

        attention1 = attention1 + attres
        attention1 = attention1.softmax(dim=-1)
        x = (attention1 @ v.transpose(-1, -2)).transpose(-1, -2)
        x = res * x
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc1D(dim=dim, k=3),
                                  ConvPosEnc1D(dim=dim, k=3)])

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentiondwconve2(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = nn.Dropout(drop_rate)
        if self.ffn:
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = FeedForward(dim, mlp_hidden_dim, dropout = drop_rate)

        self.conv1 = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding="same",groups=dim)
        self.conv2 = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=2 * self.window_size + 1,stride=1,padding="same",groups=dim)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.simgoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        B, N, C = x.shape
        shortcut = self.cpe[0](x)
        
        x = self.norm1(shortcut)
        x2 = x.permute(0,2,1)
        x2 = self.gap(x2)
        x2 = self.conv1(x2)
        x2 = self.simgoid(x2)
        x2 = x2.permute(0,2,1)
        x2 = x2.unsqueeze(1).repeat(1, N//self.window_size, 1, 1).reshape(-1, 1, C)

        x3 = self.conv2(x.permute(0,2,1))
        x3 = window_partition(x3.permute(0,2,1),self.window_size)
        x_windows = window_partition(x, self.window_size)

        attn_windows = self.attn(x_windows,x2,x3)
        x = window_reverse(attn_windows, self.window_size, N)
        x = shortcut + self.drop_path(x)
        x = self.cpe[1](x)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x

class FeedForwardFAN(nn.Module):
    def __init__(self, inputdim,outputdim, mlp_rate=2,rate=0.25,dropout = 0.):
        super().__init__()
        p_outputdim = int(outputdim * rate)
        g_outputdim = outputdim - p_outputdim * 2
        self.dense1 = nn.Linear(inputdim, p_outputdim)
        self.dense2 = nn.Linear(inputdim, g_outputdim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.dwconv = nn.Conv1d(inputdim,outputdim,kernel_size=3,stride=1,padding="same",groups=inputdim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = self.gap(x.permute(0,2,1))
        res = self.dwconv(res)
        res = self.sigmoid(res)
        res = res.permute(0,2,1)
        g = self.gelu(self.dense2(x))
        p = self.dense1(x)
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        output = output * res
        return output

class encode(nn.Module):
    def __init__(self, dim, num_heads,depth=1, window_size=7,
                 mlp_ratio=4.,drop_rate=0.):
        super().__init__()
        self.channelblock = nn.ModuleList(
            ChannelBlock(dim, num_heads,mlp_ratio=mlp_ratio,drop_rate=drop_rate)
            for i in range(depth)
        )
        self.Spatialblock = nn.ModuleList(
            SpatialBlock(dim, num_heads, window_size=window_size,mlp_ratio=mlp_ratio, drop_rate=drop_rate)
            for i in range(depth)
        )
        self.depth = depth

    def forward(self, x):
        for i in range(self.depth):
            x = self.Spatialblock[i](x)
            x = self.channelblock[i](x)
        return x

class FFT_branch(torch.nn.Module):
    def __init__(self,length):
        super(FFT_branch, self).__init__()
        self.size = length
        self.laynorm = nn.LayerNorm(normalized_shape=self.size//2 + 1)
    def forward(self,x):
        data_length = x.shape[-1]
        x1 = torch.abs(torch.fft.rfft(x)) / data_length  
        y = self.laynorm(x1)        
        return y

class TFDAVIT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, windowsize, dim, depth, heads, mlp_ratio = 4.,layers = [1], channels = 12, dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0
        self.depth = depth

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.dim = dim
        self.windowsize = windowsize
        self.patch_embedding = nn.Conv1d(in_channels=channels,out_channels=dim,stride=patch_size,kernel_size=patch_size,padding=0)

        self.encode = nn.ModuleList(
            [
                encode(dim = dim, num_heads=heads,mlp_ratio=mlp_ratio,drop_rate=dropout,window_size=windowsize,depth=depth)
                for i in range(depth)
            ]
        )

        self.fft_conv = nn.ModuleList(
            [
            nn.Sequential(
                nn.Conv1d(in_channels=dim,out_channels=dim * 2,kernel_size=5,stride=1,padding="same"),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(in_channels=dim * 2,out_channels=dim,kernel_size=1,stride=1,padding="same"),
                nn.BatchNorm1d(dim)
                )
                for i in range(depth)
            ]
        )

        self.fft_branch = nn.ModuleList(
            [
                FFT_branch(length=seq_len // patch_size)
                for i in range(depth)
            ]
        )

        self.mfc1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=101,stride=1,padding=50),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.mfc2 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=201,stride=1,padding=100),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.mfc3 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=301,stride=1,padding=150),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.bottleneck1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,out_channels=channels * 2,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels * 2,out_channels=channels * 4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(channels * 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels * 4,out_channels=channels *4,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(channels * 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels * 4,out_channels=channels * 2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),            
            nn.Conv1d(in_channels=channels * 2,out_channels=channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm1d(channels)
        )
        self.convt2 = nn.ModuleList(
            [
                nn.Conv1d(in_channels=dim*2,out_channels=dim,kernel_size=1,stride=1,padding="same")
                for i in range(depth)
            ]
        )

        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(depth)]
        )
        self.beta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(depth)]
        )
        
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mlp_head = nn.Linear(dim, num_classes)
        self.norelayer = nn.LayerNorm(dim)
        self.flatten = nn.Flatten()

        self.mfc1a = nn.Parameter(torch.zeros(1)) 
        self.mfc2a = nn.Parameter(torch.zeros(1)) 
        self.conveca = nn.Conv1d(in_channels=1,out_channels=12,kernel_size=3,stride=1,padding="same")
        self.sigmoid = nn.Sigmoid()
        self.resa = nn.Parameter(torch.ones(1)) 
        
        self.gelu = nn.GELU()
        self.fft = FFT_branch(seq_len)

        self.conv1 = nn.Conv1d(in_channels=1,out_channels=dim,kernel_size=4,stride=1,padding=1)
        self.conv2 = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=9,stride=8,padding=4)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.convffteca = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1,padding="same")
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, series):
        device = series.device
        x_fft = self.fft(series)
        x_fft = self.conv1(x_fft)
        x_fft = self.bn(x_fft)
        x_fft = self.conv2(x_fft)

        ecax = self.sigmoid(self.conveca(self.avg(series)))

        x1 = self.mfc1(series)
        x2 = self.mfc2(series + self.mfc1a * x1.mean(dim=1, keepdim=True))
        x3 = self.mfc3(series + self.mfc2a * x2.mean(dim=1, keepdim=True))
        x = torch.cat([x1, x2, x3], dim=1)
        res = x 
        x = self.bottleneck1(x)
        x = x * ecax * self.resa +  res 
        for i in range(self.depth):
            if i<=0:
                x = self.patch_embedding(x)
            res = x
            
            x_fft = self.fft_conv[i](x_fft)
            x = torch.cat([x_fft,x],dim=1)
            x = self.convt2[i](x)
            x = x.permute(0,2,1)
            x = self.encode[i](x)
            x = x.permute(0,2,1)
            A = self.alpha[i]
            B = self.beta[i] 
            x = A * x + B * res
        
        xt = self.avg(x_fft)
        xt = self.convffteca(xt)
        xt = self.sigmoid(xt)
        xt = xt * x

        x = self.avg(xt + x)
        x = self.flatten(x)
        out = self.norelayer(x)
        out = self.mlp_head(out)
        return out