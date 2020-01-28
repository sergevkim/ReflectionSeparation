import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, ReLU
from tensorflow.keras import Model


class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv_intro1 = Conv2D(3, 64, kernel_size=9, padding=4)
        self.conv_intro2 = Conv2D(64, 64, kernel_size=9, padding=4)
        self.conv_intro3 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_intro4 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_intro5 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_intro6 = Conv2D(64, 64, kernel_size=5, padding=2)

        self.conv_down1 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_down2 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_down3 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_down4 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_down5 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_down6 = Conv2D(64, 64, kernel_size=5, padding=2)

        self.conv_up1 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_up2 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_up3 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_up4 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_up5 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_up6 = Conv2D(64, 64, kernel_size=5, padding=2)

        self.conv_head11 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_head12 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_head13 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_head14 = Conv2D(64, 64, kernel_size=5, padding=2)
        self.conv_head15 = Conv2D(64, 64, kernel_size=9, padding=4)
        self.conv_head16 = Conv2D(64, 64, kernel_size=9, padding=4)

    def intro(self, x):
        x = self.conv_intro1(x)
        x = ReLU(x)
        x = self.conv_intro2(x)
        x = ReLU(x)
        x = self.conv_intro3(x)
        x = ReLU(x)
        x = self.conv_intro4(x)
        x = ReLU(x)
        x = self.conv_intro5(x)
        x = ReLU(x)
        x = self.conv_intro6(x)
        return x

    def body(self, x):
        return x

    def head1(self, x):
        x = self.conv_head1(x)
        x = ReLU(x)
        x = self.conv_head2(x)
        x = ReLU(x)
        x = self.conv_head3(x)
        x = ReLU(x)
        x = self.conv_head4(x)
        x = ReLU(x)
        x = self.conv_head5(x)
        x = ReLU(x)
        x = self.conv_head6(x)
        return x

    def call(self, x):
        x = self.intro(x)
        x = self.body(x)
        x = self.head1(x)
        return x



