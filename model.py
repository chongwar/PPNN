import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

class PPNN(nn.Module):
    def __init__(self, num_classes, F_t=8, F_s=16, T=256, C=64, drop_out=0.5):
        super(PPNN, self).__init__()
        self.F_t = F_t
        self.F_s = F_s
        self.T = T
        self.C = C
        self.drop_out = drop_out

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )

        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(self.F_t),
            nn.ELU(),
            # nn.AvgPool2d((1, 8)),
            # nn.Dropout(self.drop_out)
        )


        self.block_3 = nn.Sequential(
            nn.Conv2d(self.F_t, self.F_s, (self.C, 1)),
            nn.BatchNorm2d(self.F_s),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

        self.fc = nn.Linear(self.F_s * self.T , num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
    
if __name__ == '__main__':
    pass
