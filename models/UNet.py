import copy

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

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
        self.conv_down_5_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_down_5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_down_6_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_down_6_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_up_1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up_2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up_2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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
        legacy_1 = self.conv_down_1(x)
        legacy_1 = F.relu(legacy_1)
        legacy_1 = self.conv_down_2(legacy_1)
        legacy_1 = F.relu(legacy_1)   # shape is 32
        legacy_2 = self.conv_down_3(legacy_1)
        legacy_2 = F.relu(legacy_2)
        legacy_2 = self.conv_down_4(legacy_2)
        legacy_2 = F.relu(legacy_2)   # shape is 64
        legacy_3 = self.conv_down_5_1(legacy_2)
        legacy_3 = self.conv_down_5_2(legacy_3)
        legacy_3 = F.relu(legacy_3)
        legacy_3 = self.conv_down_6_1(legacy_3)
        legacy_3 = self.conv_down_6_2(legacy_3)
        legacy_3 = F.relu(legacy_3)

        up = self.conv_up_1_1(legacy_3)
        up = self.conv_up_1_2(up)
        up = F.relu(up)
        up = self.conv_up_2_1(up)
        up = self.conv_up_2_2(up)
        up = F.relu(up)
        up = self.conv_up_3(legacy_2 + up)
        up = F.relu(up)
        up = self.conv_up_4(up)
        up = F.relu(up)
        up = self.conv_up_5(legacy_1 + up)
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

        output = {
            'transmission': transmission,
            'reflection': reflection
        }

        return output


    def prepare_batch(
            self,
            subject,
            astigma,
            device,
            epoch,
            reflection_kernel_size=(9, 9),
            blur_kernel_size=(5, 5),
            train=True):

        if train:
            if epoch < 2:
                alpha = np.random.uniform(0.75, 0.8)
            else:
                alpha = np.random.uniform(0.4, 0.8)
        else:
            alpha = np.random.uniform(0.6, 0.8)

        reflection_kernel = np.zeros(reflection_kernel_size)
        x1, y1, x2, y2 = np.random.randint(0, reflection_kernel_size[0], size=4)
        reflection_kernel[x1, y1] = 1.0 - np.sqrt(alpha)
        reflection_kernel[x2, y2] = np.sqrt(alpha) - alpha
        reflection_kernel = cv2.GaussianBlur(reflection_kernel, blur_kernel_size, 0)

        reflection_kernel_2 = np.array([
            np.array([reflection_kernel, np.zeros_like(reflection_kernel), np.zeros_like(reflection_kernel)]),
            np.array([np.zeros_like(reflection_kernel), reflection_kernel, np.zeros_like(reflection_kernel)]),
            np.array([np.zeros_like(reflection_kernel), np.zeros_like(reflection_kernel), reflection_kernel]),
        ])
        reflection_kernel_2 = torch.Tensor(reflection_kernel_2)

        reflection = F.conv2d(
            astigma,
            reflection_kernel_2,
            padding=reflection_kernel_size[0] // 2)

        transmission = copy.deepcopy(subject)
        transmission[:, 0, :, :] * alpha

        synthetic = transmission + reflection

        output = {
            'synthetic': synthetic.to(device),
            'transmission': transmission.to(device),
            'reflection': reflection.to(device),
        }

        return output


    def compute_losses(self, batch):
        output = self.forward(batch['synthetic'])

        loss_transmission = F.mse_loss(output['transmission'], batch['transmission'])
        loss_reflection = F.mse_loss(output['reflection'], batch['reflection'])
        loss = loss_transmission + loss_reflection

        #TODO utils.scripts or other?
        '''
        scripts.save(output['trans'], 'imgs')
        scripts.save(batch['synthetic'], 'syns')
        scripts.save(batch['alpha_transmitted'], 'alphas')
        scripts.save(batch['reflected'], 'refs')
        '''
        #TODO: add VGG L2
        losses = {
            'full': loss,
            'transmission': loss_transmission,
            'reflection': loss_reflection
        }

        return losses

