import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)#递归创建目录
    '''
    makedirs()方法是递归目录创建功能。如果exists_ok为False(默认值)，
    则如果目标目录已存在，则引发OSError错误，True则不会
    '''
    #当我们执行某个 Python 代码，例如文件 mycode.py 时，想要传递一些可以随时改变的自定义的参数。
    # 比如在训练神经网络的时候，我们为了方便修改训练的 batch 大小，epoch 的大小等等，往往不想去动代码。
    # 此时最方便的方法就是在执行代码的时候从命令行传入参数。argparse.ArgumentParser() 可以很好地满足这一需求。
    #help - 一个此选项作用的简单描述
    parser = argparse.ArgumentParser()#创建解析器使用 argparse 的第一步是创建一个 ArgumentParser 对象
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # 生成原始噪点数据大小--latent_dim
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()#args = parser.parse_args() 则是使得改代码生效
    print(opt)
    
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    # 默认：通道数=1，图片大小=28*28
    # 这些参数opt.channels, opt.img_size, opt.img_size便是需要去上一部分设定的参数的位置去找的，
    # 都是带有opt. 意思为图像的通道数为1，尺寸大小为28*28，通道数为1表示是灰度图
    cuda = True if torch.cuda.is_available() else False
    # 使用GPU的语句，有GPU就可以使用GPU运算

    # 定义；两个模块，一个生成器，一个判别器。
    # 定义生成器类   原来的是1×28×28   生成一个数据相当于是mnist数据
    '''
    这一部分代码是搭建生成器神经网络，对于小白就当成一个套路来做，就是每次搭建网络都这样写，
    只是改变一下*block里面的数字和激活函数来测试就行，等一段时间学懂了在自己变换神经元的层数和神经层。
    至于forward中的z是在程序后面的定义的高斯噪声信号，形状为64*100，所以如果你非要问img.size(0)的话，
    它为64，也就是一批次训练的数目。
    '''
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
    
            def block(in_feat, out_feat, normalize=True):#定义一个静态方法，方便搭建网络
                layers = [nn.Linear(in_feat, out_feat)]# 对传入数据应用线性转换（输入节点数，输出节点数）
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))# 批规范化
                layers.append(nn.LeakyReLU(0.2, inplace=True))  # 激活函数
                return layers
    
            self.model = nn.Sequential(
                # 先从一堆随机值，100个特征-通过网络
                # ->变为需要的生成数据(要和判别数据一样的形状大小)
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                # 最后得到1024个特征，1×28×28=724个像素点，现在得到的特征个数需要跟原始的一致才可以
                nn.Tanh()#对数据进行激活,将元素调整到区间(-1,1)内
            ) #快速搭建网络， np.prod 用来计算所有元素的乘积
    
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *img_shape)
            #view()函数是用于对Tensor(张量)进行形状变化的函数,如一个Tensor的size是3x2,
            # 可以通过view()函数将其形状转为2x3
            return img
    
    
    # 定义判别器类  判断一张图片是真是假
    # 输入实际的图像724像素点 经过几个全连接和激活函数，得到预测值。再将预测值传入sigmoid当中，
    # 接着可以传入BCELoss算损失
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
    
            self.model = nn.Sequential(  # 输入图像是1×28×28=724个像素点
                nn.Linear(int(np.prod(img_shape)), 512),  # 得到512
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),  # 最后得到一个预测值
                nn.Sigmoid(),
            )
    
        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            #在pytorch中的view()函数就是用来改变tensor的形状的，
            # 例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
            validity = self.model(img_flat)
            return validity
    
    
    # 1.Loss function
    adversarial_loss = torch.nn.BCELoss()
    #定义了一个损失函数nn.BCELoss()，输入（X，Y）, X 需要经过sigmoid,
    # Y元素的值只能是0或1的float值，依据这个损失函数来计算损失
    
    # 2.Initialize generator and discriminator  生成器和判别器
    # 这部分是初始化生成器和鉴别器
    generator = Generator()
    discriminator = Discriminator()
    
    # 用GPU来训练，if cuda表示如果可以进行GPU加速，则在GPU内建立生成器、鉴别器和损失函数，
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    dataroot = "G:\science_data\datasets\RicePestv3_category_train"
    workers = 16
    dataset = datasets.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.img_size),
                                transforms.CenterCrop(opt.img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=workers)

    # Optimizers  指定带动量的优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # ----------
    #  Training
    # ----------
    # 训练时迭代每个epochs
    for epoch in range(opt.n_epochs):
        #dataloader中的数据是一张图片对应一个标签，所以imgs对应的是图片，_对应的是标签，
        # 而i是enumerate输出的功能，代表序号，enumerate用于将一个可遍历的数据对象(如列表、元组或字符串)
        # 组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中，所以i就是相当于1,2,3…..的数据下标。
        for i, (imgs, _) in enumerate(dataloader):
    
            # Adversarial ground truth 两个标签:真的/假的  对判别器来说真的数据是mnist中的数据标签为为1，假数据是生成器生成的图像标签为0；
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  # 填充1和0标签
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
    
            # Configure input拿到实际的数据
            # 这句是将真实的图片转化为神经网络可以处理的变量。变为Tensor
            real_imgs = Variable(imgs.type(Tensor))
    
            # -----------------
            #  Train Generator
            # -----------------
    
            # 首先梯度清零
            # 每次的训练之前都将上一次的梯度置为零，以避免上一次的梯度的干扰
            optimizer_G.zero_grad()
    
            # Sample noise as generator input
            # 根据batch数，随机构建一个100维的向量    初始化随机的一个batch的向量
            # 这部分就是在上面训练生成网络的z的输入值，
            # np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)的意思就是
            # 64个噪音（基础值为100大小的） 0，代表正态分布的均值，1，代表正态分布的方差
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images     通过生成器把 100维的向量生成一个784的特征
            gen_imgs = generator(z)
    
            # Loss measures generator's ability to fool the discriminator
            # 将生成的数据传入判别器，传入标签值valid为1，我们想让生成器骗过判别器，所以计算生成器的损失。
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    
    
            # 反向传播
            # 更新梯度
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # 梯度清零
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            # 传入实际数据，判别器希望判别出来是真，所以标签值为1.# 判别器判别真实图片是真的的损失
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            # 传入生成数据，判别器希望判别为假，所以标签值为0 # 判别器判别假的图片是假的的损失
            d_loss = (real_loss + fake_loss) / 2  #总的误差取平均
            # 计算损失值，优化更新梯度
            d_loss.backward()
            optimizer_D.step()
    
            # 打印，保存数据
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
    
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)