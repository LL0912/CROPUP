#channel-time attention network
import torch
import torch.nn as nn
from models.layers import LinearLayer, get_positional_encoding
from models.decoder import get_decoder
from copy import deepcopy
import numpy as np
from thop import profile

class unite(nn.Module):
    def __init__(self,
                 input_dim=10,
                 mlp1_m=[64,128,256],
                 ca_ratio=16,
                 n_head=16,
                 d_model=256,
                 d_k=8,
                 mlp1_t=[256, 128],
                 dropout=0.2,
                 T=1000,
                 max_temporal_shift=30,
                 max_position=365,
                 atten_type="cross",
                 ch_type="att",
                 mlp_cls=[128,64,32],
                 num_classes=10,
                 ):
        """
        Args:
            in_channel:

        """
        super(unite,self).__init__()
        self.ch_type=ch_type
        mlp1 = []
        mlp1.append(input_dim)
        mlp1.extend(mlp1_m)

        self.channel_enc=MLP_Encoder(mlp1)

        self.channel_attention=Channel_attention(mlp1[-1],
                                                 ca_ratio)
        self.time_attention=Time_attention(in_channels=mlp1[-1],
                                           n_head=n_head,
                                           d_k=d_k,
                                           n_neurons=mlp1_t,
                                           dropout=dropout,
                                           d_model=d_model,
                                           T=T,
                                           max_temporal_shift=max_temporal_shift,
                                           max_position=max_position,
                                           atten_type=atten_type
                                           )

        self.decoder = get_decoder(mlp_cls, num_classes)
        self.param_ratio()

    def forward(self, x, positions, return_feats=False):
        x=x.squeeze(-1)
        out=self.channel_enc(x)
        if self.ch_type == "att":
            out=self.channel_attention(out)
        temporal_feats = self.time_attention(out, positions)
        logits = self.decoder(temporal_feats)
        if return_feats:
            return logits, temporal_feats
        else:
            return logits
    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.channel_attention)
        t = get_ntrainparams(self.time_attention)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                s / total * 100, t / total * 100, c / total * 100
            )
        )

        return total
class MLP_Encoder(nn.Module):
    def __init__(
            self,
            mlp1,
    ):
        super(MLP_Encoder, self).__init__()
        self.mlp1_dim = deepcopy(mlp1)
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(LinearLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)
    def forward(self, x):
        out = x  #B,T,C
        out = self.mlp1(out)
        return out  # B,T,C
class Channel_attention(nn.Module):
    def __init__(self,
                 in_channel,
                 reduction_ratio,
                 ):
        super(Channel_attention,self).__init__()
        self.avgtime=nn.AdaptiveAvgPool1d(1)
        self.channel_se=nn.Sequential(nn.Linear(in_channel,in_channel//reduction_ratio,bias=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(in_channel // reduction_ratio, in_channel, bias=False),
                                      nn.Sigmoid()
                                      )

    def forward(self,x):
        """
        input x: [B,T,C]
        """
        residual=x
        out=x.permute(0,2,1)
        out=self.avgtime(out)
        out=out.permute(0,2,1)
        att=self.channel_se(out)
        out=att*out
        out=out+residual
        return out


class Time_attention(nn.Module):
    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=8,
                 n_neurons=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 max_temporal_shift=100,
                 max_position=365,
                 atten_type="cross"
                 ):
        """
        """

        super(Time_attention, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = deepcopy(n_neurons)
        self.max_temporal_shift = max_temporal_shift

        if d_model is not None:
            self.d_model = d_model
            self.inconv = LinearLayer(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.positional_enc = nn.Embedding.from_pretrained(get_positional_encoding(max_position + 2*max_temporal_shift, self.d_model, T=T), freeze=True)

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model,atten_type=atten_type)

        assert (self.n_neurons[0] == self.d_model)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(LinearLayer(self.n_neurons[i], self.n_neurons[i + 1]))

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, return_att=False):
        if self.inconv is not None:
            x = self.inconv(x)
        enc_output = x + self.positional_enc(positions)

        enc_output, attn = self.attention_heads(enc_output)

        enc_output = self.dropout(self.mlp(enc_output))

        if return_att:
            return enc_output, attn
        else:
            return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_k, d_in, atten_type):
        """
        atten_type:["light","cross"]
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.atten_type=atten_type
        self.key = nn.Linear(d_in, n_head * d_k)
        if self.atten_type=="light":
            self.query = nn.Parameter(torch.zeros(n_head, d_k)).requires_grad_(True)
            nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / (d_k)))
        if self.atten_type=="cross":
            self.query = nn.Linear(d_in, n_head * d_k)
            #nn.init.normal_(self.query.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        if self.atten_type=="tae":
            self.query = nn.Linear(d_in, n_head * d_k)
            nn.init.normal_(self.query.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
            self.fc2 = nn.Sequential(nn.BatchNorm1d(n_head * d_k),
                                     nn.Linear(n_head * d_k, n_head * d_k))
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B, T, C = x.size()
        if self.atten_type=="light":
            q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, 1, d_k)
            k = self.key(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, d_k)
            v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # self-attend; (B, nh, 1, d_k) x (B, nh, d_k, T) -> (B, nh, 1, T)
            # q=torch.mean(k,dim=2).unsqueeze(dim=2) # (B, nh, 1, d_k)
            att = (q @ k.transpose(-2, -1)) / self.temperature
            att = self.softmax(att)
            att = self.dropout(att)
            y = att @ v  # (B, nh, 1, T) x (B, nh, T, hs) -> (B, nh, 1, hs)
            y = y.transpose(1, 2).contiguous().view(B, C)
            return y, att

        if self.atten_type=="tae":
            q = self.query(x).view(B, T, self.n_head, self.d_k)
            q = q.mean(dim=1).squeeze()  # MEAN query
            q = self.fc2(q.view(B, self.n_head * self.d_k)).view(B, self.n_head, self.d_k)
            q = q.permute(1, 0, 2).contiguous().view(self.n_head * B, self.d_k)
            k = self.key(x).view(B, T, self.n_head, self.d_k)
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, T, self.d_k)  # (n*b) x lk x dk
            v = x.repeat(self.n_head, 1, 1)  # (n*b) x lv x d_in
            output, attn = self.attention(q, k, v)
            output = output.view(self.n_head, B, 1, C)
            output = output.squeeze(dim=2)
            attn = attn.view(self.n_head, B, 1, T)
            attn = attn.squeeze(dim=2)
            return output, attn

        if self.atten_type=="cross":
            k = self.key(x).view(B, T, self.n_head, self.d_k).permute(0,2,3,1) # [B,n_h,d_k,T]
            q = self.query(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # [B,n_h,T,d_k]
            v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            att=(q@k)/self.temperature #[B,N_H,T,T]
            att = torch.mean(att,dim=-1)
            # att=torch.mean(torch.cat([torch.mean(att,-1).unsqueeze(-1),torch.mean(att,-2).unsqueeze(-2).transpose(-2,-1)],dim=-1),dim=-1)
            att = self.softmax(att)
            att = self.dropout(att).unsqueeze(2)
            y = att @ v  # (B, nh, 1, T) x (B, nh, T, hs) -> (B, nh, 1, hs)
            y = y.transpose(1, 2).contiguous().view(B, C)
            return y, att
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__=="__main__":
    model=unite(input_dim=10,num_classes=7)
    batch_size=1024
    x=torch.randn((batch_size,9,10))
    position=torch.randint(0,365,(batch_size,9))
    mask=None
    extra=None
    out=model(x,position)
    # print()
    # print(out.shape)
    # criterion = nn.CrossEntropyLoss()
    # targets=torch.empty(batch_size).random_(2).long()
    # loss = criterion(out, targets)

    flops, params = profile(model, inputs=(x,position))
    print(flops/1e9,params/1e6) #flops单位G，para单位M