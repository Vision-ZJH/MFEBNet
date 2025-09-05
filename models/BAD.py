import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class BAD(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(BAD, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        out = F.avg_pool2d(x, kernel_size=[h, w]).view(b, -1)

        query = out
        key = out
        value = out
        energy = torch.bmm(query.unsqueeze(2), key.unsqueeze(1))
        query_key = energy.max(dim=-1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(query_key)
        out = torch.bmm(attention, value.unsqueeze(-1)).squeeze(-1)
        out = self.dense(out)
        out = x * out.view(b, c, 1, 1)
        return out