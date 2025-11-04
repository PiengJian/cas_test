import torch
import torch.nn as nn

class RelativeL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def rel_loss(self, x, y):
        batch_size = x.shape[0]
        x_ = x.reshape((batch_size, -1))
        y_ = y.reshape((batch_size, -1))

        
        diff_norms = torch.norm(x_ - y_, p=2,  dim=1)
        y_norms = torch.norm(y_, p=2,  dim=1)
        return diff_norms / y_norms
    
    
    def forward(self, output, label, weight_dict=None):
        losses = 0
        for i in range(len(label)):
            loss = self.rel_loss(output[i], label[i])
            if weight_dict is not None:
                loss *= weight_dict[i]

            loss = loss.mean()
            losses += loss
        return losses
    
    

class RelativeL2LossForC(nn.Module):
    def __init__(self):
        super().__init__()

    def rel_loss_per_variable(self, x, y):
        # x, y: (B, 1, H, W)
        batch_size = x.shape[0]
        x_ = x.reshape(batch_size, -1)  # (B, H*W)
        y_ = y.reshape(batch_size, -1)  # (B, H*W)
        diff_norms = torch.norm(x_ - y_, p=2, dim=1)  # (B,)
        y_norms = torch.norm(y_, p=2, dim=1)          # (B,)
        return diff_norms / y_norms                    # (B,)

    def forward(self, output, label, variable_weights=None, time_weights=None):
        # output, label: 长度为 6 的列表，每个元素形状为 (B, 24, H, W)
        # variable_weights: 形状为 (24,) 的张量或列表，为每个变量指定权重
        # time_weights: 长度为 6 的张量或列表，为每个时间步指定权重（可选）
        losses = 0
        for i in range(len(label)):
            B, C, H, W = output[i].shape  # C = 24
            loss_i = 0
            for j in range(C):  # 遍历 24 个变量
                x_j = output[i][:, j:j+1, :, :]  # (B, 1, H, W)
                y_j = label[i][:, j:j+1, :, :]   # (B, 1, H, W)
                loss_j = self.rel_loss_per_variable(x_j, y_j)  # (B,)
                if variable_weights is not None:
                    loss_j = loss_j * variable_weights[j]  # 对变量 j 应用权重
                loss_i = loss_i + loss_j.mean()  # 对批量维度取平均并累加
            if time_weights is not None:
                loss_i = loss_i * time_weights[i]  # 对时间步 i 应用权重
            losses = losses + loss_i
        return losses
















    # def forward(self, x, y):
    #     batch_size = x.shape[0]
    #     x_ = x.reshape((batch_size, -1))
    #     y_ = y.reshape((batch_size, -1))
       
    #     # 计算差异的L2范数
    #     diff_norm = torch.norm(x_ - y_, p=2, dim=1)

    #     # 计算y的L2范数
    #     y_norm = torch.norm(y_, p=2)

    #     # 计算相对L2损失
    #     loss = diff_norm / y_norm

    #     loss = loss.mean()

    #     return loss
