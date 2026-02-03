import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.B = nn.Parameter(torch.randn(num_nodes, num_nodes) * 1e-5, requires_grad=True)

    def forward(self, x):
        N, C, T, V = x.size()
        A = torch.eye(V).to(x.device) + self.B
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(-1, V)
        x_graph = torch.matmul(x_reshaped, A)
        x_graph = x_graph.view(N, T, C, V).permute(0, 2, 1, 3)
        return self.conv(x_graph)

class AdaptiveSTGCN(nn.Module):
    def __init__(self, num_classes=2, num_joints=33, T=100):
        super().__init__()
        self.gcn1 = AdaptiveGraphConv(3, 64, num_joints)
        self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)

        self.gcn2 = AdaptiveGraphConv(64, 128, num_joints)
        self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)

        self.gcn3 = AdaptiveGraphConv(128, 256, num_joints)
        self.tcn3 = nn.Conv2d(256, 256, kernel_size=(9, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.tcn1(self.gcn1(x))))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.bn2(self.tcn2(self.gcn2(x))))
        x = F.dropout(x, 0.4, training=self.training)
        x = F.relu(self.bn3(self.tcn3(self.gcn3(x))))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.fc(x)