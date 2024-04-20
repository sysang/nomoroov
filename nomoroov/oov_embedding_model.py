import torch
from torch import nn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class OovEmbbeding(nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.dim_size  = dim_size
        self.twin1 = torch.nn.Parameter(torch.ones(dim_size, dtype=torch.float32), requires_grad=True)
        self.twin2 = torch.nn.Parameter(torch.zeros(dim_size, dtype=torch.float32), requires_grad=True)

    def forward(self, x1, length1, x2, length2):
        masking_ratio = 0.83
        noise_ratio = 0.67

        masking1 = torch.rand(self.dim_size, dtype=torch.float32) < masking_ratio
        masking1 = masking1.int()
        masked_twin1 =  self.twin1.mul(masking1)

        masking2 = torch.rand(self.dim_size, dtype=torch.float32) < masking_ratio 
        masking2 = masking2.int()
        masked_twin2 = self.twin2.mul(masking2)

        aver_x1_x2 = x1.add(x2).div(2)
        noise1 = torch.rand(self.dim_size).mul(noise_ratio).mul(aver_x1_x2)
        noise2 = torch.rand(self.dim_size).mul(noise_ratio).mul(aver_x1_x2)

        result_x1_twin1 = x1.add(masked_twin1).add(noise1).div(length1)
        result_x1_twin2 = x1.add(masked_twin2).add(noise2).div(length1)
        result_x2_twin1 = x2.add(masked_twin1).add(noise1).div(length2)
        result_x2_twin2 = x2.add(masked_twin2).add(noise2).div(length2)

        distance1 = cos(result_x1_twin1, result_x2_twin2)
        distance2 = cos(result_x1_twin2, result_x2_twin1)
        add_twin1 = result_x1_twin1.add(result_x2_twin1)
        add_twin2 = result_x1_twin2.add(result_x2_twin2)
        diff1 = distance1.sub(distance2).abs()
        diff2 = cos(result_x1_twin1, result_x1_twin2).sub(1).abs()
        diff3 = cos(result_x2_twin1, result_x2_twin2).sub(1).abs()
        diff4 = cos(add_twin1, add_twin2).sub(1).abs()
        diff = diff1.add(diff2).add(diff3).add(diff4)

        return diff
