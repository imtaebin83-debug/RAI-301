import torch
from torch import nn
from torch.nn import functional as F


def rotvec_to_sixd(rotvec):
    """Continuous rotation features, including the +/-pi rotvec branch."""
    theta=torch.linalg.vector_norm(rotvec,dim=-1,keepdim=True)
    x,y,z=rotvec.unbind(-1)
    zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*rotvec.shape[:-1],3,3)
    a=torch.sinc(theta/torch.pi)[...,None]
    b=(.5*torch.sinc(theta/(2*torch.pi)).square())[...,None]
    matrix=torch.eye(3,dtype=rotvec.dtype,device=rotvec.device)+a*skew+b*(skew@skew)
    return matrix[...,:,:2].flatten(-2)


class SpatialSummary(nn.Module):
    """Keep the position of visual features instead of discarding it by pooling."""
    def forward(self, features):
        b,c,h,w=features.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=features.device,dtype=features.dtype),
                             torch.linspace(-1,1,w,device=features.device,dtype=features.dtype),indexing='ij')
        weights=(features.flatten(2)/.2).softmax(-1)
        coords=torch.stack(((weights*xx.flatten()).sum(-1),(weights*yy.flatten()).sum(-1)),dim=-1).flatten(1)
        return torch.cat((coords,features.mean((-2,-1))),dim=-1)


class TwoCameraBC(nn.Module):
    def __init__(self, horizon=8, image_size=128):
        super().__init__()
        self.horizon, self.image_size = horizon, image_size
        self.register_buffer('input_mean',torch.zeros(8))
        self.register_buffer('input_std',torch.ones(8))
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            SpatialSummary(), nn.Linear(64 * 3, 128), nn.ReLU())
        self.proprio = nn.Sequential(nn.Linear(11, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(320, 256), nn.ReLU(),
                                  nn.Linear(256, horizon * 7))

    def set_state_normalization(self,mean,std):
        self.input_mean.copy_(torch.as_tensor(mean,device=self.input_mean.device))
        self.input_std.copy_(torch.as_tensor(std,device=self.input_std.device))

    def forward(self, images, state):
        b = images.shape[0]
        x = images.reshape(b * 2, 3, *images.shape[-2:])
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        visual = self.visual(x).reshape(b, 256)
        raw_state=state*self.input_std+self.input_mean
        proprio=torch.cat((state[:,:3],rotvec_to_sixd(raw_state[:,3:6]),state[:,6:]),dim=-1)
        return self.head(torch.cat((visual, self.proprio(proprio)), -1)).reshape(b, self.horizon, 7)


def bc_loss(output, actions, mask):
    motion = F.smooth_l1_loss(output[..., :6].tanh(), actions[..., :6], reduction="none").mean(-1)
    grip = F.binary_cross_entropy_with_logits(output[..., 6], (actions[..., 6] >= 0).float(), reduction="none")
    weight = mask.to(output.dtype)
    return ((motion + .2 * grip) * weight).sum() / weight.sum().clamp_min(1)


def decode_actions(output):
    return torch.cat((output[..., :6].tanh(), torch.where(output[..., 6:] >= 0, 1., -1.)), -1)
