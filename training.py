import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from agents import Generator, Discriminator, compute_gradient_penalty, Tensor
from hyperparameters import batch_size, lr, beta1, beta2, n_epochs, latent_dim, lambda_gp, n_critic, sample_interval

cuda = True if torch.cuda.is_available() else False

# init agents
generator = Generator()
discriminator = Discriminator()

# move to gpu
if cuda:
    generator.cuda()
    discriminator.cuda()

# data set and loader
dataset = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

batches_done = 0
for epoch in range(n_epochs):
    print("running epoch ", epoch)
    for i, (imgs, _) in enumerate(dataloader):
        # this loop trains discrim

        # config input
        real_imgs = imgs.type(Tensor)

        optimizer_D.zero_grad()

        # input a bunch of bullshit into the generator for fake images
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        fake_imgs = generator(z)

        # input images into discrim
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        # gradient penalty
        grad_pen = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        # loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * grad_pen
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # train generator every n_critic steps
        if i % n_critic == 0:
            # this loop trains the generator

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            if batches_done % sample_interval == 0:
                print("saving batch", batches_done)
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += n_critic

torch.save(generator, "generator.pt")
torch.save(discriminator, "discriminator.pt")