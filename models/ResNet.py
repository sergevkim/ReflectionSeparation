import torch
import torch.nn as nn
import torch.functional as F


class ResBlock():
    def __init__(self):
        pass


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv_intro_1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv_intro_2 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.conv_intro_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_intro_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_intro_5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_intro_6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.conv_down_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_down_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2) # 32 to concat
        self.conv_down_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_down_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2) # 64 to concat
        self.conv_down_5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_down_6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.conv_up_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_up_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_up_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2) # not 64. but 64 + 64 = 128 because of concat
        self.conv_up_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_up_5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)  # not 32. but 32 + 32 = 64 because of concat
        self.conv_up_6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.conv_head_1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_1_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_1_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_1_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_1_5 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.conv_head_1_6 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

        self.conv_head_2_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_2_3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_2_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv_head_2_5 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.conv_head_2_6 = nn.Conv2d(64, 3, kernel_size=9, padding=4)


    def intro(self, x):
        x = self.conv_intro_1(x)
        x = F.relu(x)
        x = self.conv_intro_2(x)
        x = F.relu(x)
        x = self.conv_intro_3(x)
        x = F.relu(x)
        x = self.conv_intro_4(x)
        x = F.relu(x)
        x = self.conv_intro_5(x)
        x = F.relu(x)
        x = self.conv_intro_6(x)
        x = F.relu(x)

        return x


    def body(self, x):
        legacy1 = self.conv_down_1(x)
        legacy1 = F.relu(legacy1)
        legacy1 = self.conv_down_2(legacy1)
        legacy1 = F.relu(legacy1)   # shape is 32
        legacy2 = self.conv_down_3(legacy1)
        legacy2 = F.relu(legacy2)
        legacy2 = self.conv_down_4(legacy2)
        legacy2 = F.relu(legacy2)   # shape is 64
        legacy3 = self.conv_down_5(legacy2)
        legacy3 = F.relu(legacy3)
        legacy3 = self.conv_down_6(legacy3)
        legacy3 = F.relu(legacy3)   # shape is 128 (actually we needn't legacy3)
        up = self.conv_up_1(legacy3)
        up = F.relu(up)
        up = self.conv_up_2(up)
        up = F.relu(up)
        up = self.conv_up_3(legacy2 + up)
        up = F.relu(up)
        up = self.conv_up_4(up)
        up = F.relu(up)
        up = self.conv_up_5(legacy1 + up)
        up = F.relu(up)
        up = self.conv_up_6(up)
        up = F.relu(x - up)

        return up


    def head_1(self, x):
        x = self.conv_head_1_1(x)
        x = F.relu(x)
        x = self.conv_head_1_2(x)
        x = F.relu(x)
        x = self.conv_head_1_3(x)
        x = F.relu(x)
        x = self.conv_head_1_4(x)
        x = F.relu(x)
        x = self.conv_head_1_5(x)
        x = F.relu(x)
        x = self.conv_head_1_6(x)

        return x


    def head_2(self, x):
        x = self.conv_head_2_1(x)
        x = F.relu(x)
        x = self.conv_head_2_2(x)
        x = F.relu(x)
        x = self.conv_head_2_3(x)
        x = F.relu(x)
        x = self.conv_head_2_4(x)
        x = F.relu(x)
        x = self.conv_head_2_5(x)
        x = F.relu(x)
        x = self.conv_head_2_6(x)

        return x


    def forward(self, x):
        intro_output = self.intro(x)
        body_output = self.body(intro_output)
        transmission = self.head_1(body_output)
        reflection = self.head_2(body_output)

        return dict(transmission=transmission)


    def compute_all(self, batch, device=None):
        synthetic = batch['synthetic'].to(device)
        alpha_transmitted = batch['alpha_transmitted'].to(device)
        reflected = batch['reflected'].to(device)
        output = self.forward(synthetic)

        loss_trans = F.mse_loss(output['trans'], alpha_transmitted)
        #TODO loss_refl = F.mse_loss(output['reflect'], reflected)

        #loss = loss_refl + loss_trans
        loss = loss_trans

        #TODO utils.scripts or other?
        scripts.save(output['trans'], 'imgs')
        scripts.save(batch['synthetic'], 'syns')
        scripts.save(batch['alpha_transmitted'], 'alphas')
        scripts.save(batch['reflected'], 'refs')
        # todo: add VGG L2
        return dict(
            loss=loss,
            metrics=dict(
                mse_trans=loss_trans.item(),
                #mse_refl=loss_refl.item(),
            )
        )

