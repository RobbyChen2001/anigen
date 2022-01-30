import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch import optim
from torchvision import transforms
from math import ceil, log2
from scipy.stats import truncnorm
from pytorch_lightning.loggers import WandbLogger 
import os
import gdown
from torchvision.datasets import ImageFolder

root_dir = '/mnt/d/Datasets/anime-faces'
AVAIL_GPUS = 1
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_images = 21551 # Change to use automatic calculation
initial_resolution = 4
full_resolution = 32
plt.rcParams["figure.figsize"] = (12,12)

def show_tensor_images(image_tensor, rows, name = None, mean = None, std = None):
    size = image_tensor.shape[1:]
    num_images = image_tensor.shape[0]
    if std is not None and mean is not None:
        
        image_tensor = torch.tensor(std).view(size[0], 1, 1).to(device) * image_tensor + torch.tensor(mean).view(size[0], 1, 1).to(device)
    else:
        image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=rows, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    if name:
        plt.savefig(name)
    else:
        plt.show()

def get_num_steps():
    assert log2(full_resolution) == int(log2(full_resolution)), "Please use a power of 2 as the resolution"
    return int(log2(full_resolution / initial_resolution))

class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir, file_name = 'dataset.pt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Accessing dataset as tensor is much faster than using the filesystem
        if not root_dir:
            self.images = torch.load(f'{file_name}')
        else:
            self.images = torch.load(f'{root_dir}/{file_name}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img
    
    def set_transform(self, transform):
        self.transform = transform

class AnimeFaceDataModule(pl.LightningDataModule):
    def __init__(self, resolution = 64, data_dir: str = root_dir, batch_size: int = BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean, self.std = (0.7007, 0.6006, 0.5895), (0.2938, 0.2973, 0.2702)
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.Normalize(self.mean, self.std) # should we recompute mean and std for each resolution
        ])
        self.data = None

        if os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            link = 'https://drive.google.com/uc?id=1vcsXeKRT77CtoI-DAaTvUEsjfEPRRIuY'
            gdown.download(link, 'dataset.pt', False)
            self.data_dir = ''

    def __len__(self):
        return len(self.data)
    
    def setup(self, stage=None):
        self.data = AnimeFaceDataset(self.data_dir, transform = self.transform)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def set_transform(self, transform):
        self.data.set_transform(transform)

    def get_data_mean_std(self):
        return self.mean, self.std

dm = AnimeFaceDataModule()

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2,
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features, gain=2,
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)

class MappingLayers(nn.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    # Use 3 for fast calculation (8 in original paper)
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(z_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            WSLinear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)

class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        # You use nn.Parameter so that these weights can be optimized
        self.weight = nn.Parameter( 
            torch.randn(1, channels, 1, 1)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        n_samples, _, width, height = image.size()
        noise_shape = (n_samples, 1, width, height)
        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel

class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = WSLinear(w_dim, channels)
        self.style_shift_transform = WSLinear(w_dim, channels)

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image
    
    def get_style_scale_transform(self):
        return self.style_scale_transform

    def get_style_shift_transform(self):
        return self.style_shift_transform

class StyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, use_upsample=True):
        super().__init__()
        self.conv1 = WSConv2d(in_chan, out_chan, kernel_size, padding=1) # Padding is used to maintain the image size
        self.conv2 = WSConv2d(in_chan, out_chan, kernel_size, padding=1)
        self.inject_noise1 = InjectNoise(out_chan)
        self.inject_noise2 = InjectNoise(out_chan)
        self.adain1 = AdaIN(out_chan, w_dim)
        self.adain2 = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)
    
    # Original model has more layers in each generator block
    def forward(self, x, w):
        '''
        Function for completing a forward pass of StyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        x = self.adain1(self.activation(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.activation(self.inject_noise2(self.conv2(x))), w) 
        return x

def get_gradient(crit, real, fake, epsilon, alpha, current_layer):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images, alpha, current_layer)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

class StyleGANGenerator(nn.Module):
    '''
    StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''
    def __init__(self,
                 z_dim,
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 kernel_size,
                 hidden_chan,
                 img_channels = 3):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)

        self.starting_constant = nn.Parameter(torch.ones(1, in_chan, 4, 4))
        # self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.initial_block = StyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size)
    
        self.initial_rgb = WSConv2d(
            in_chan, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )
        
        for _ in range(get_num_steps()):
            self.prog_blocks.append(
                StyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size)
            )
            self.rgb_layers.append(
                WSConv2d(hidden_chan, img_channels, kernel_size=1, stride=1, padding=0)
            )
        
    def forward(self, noise, alpha, current_layer):
        '''
        Function for completing a forward pass of StyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''

        x = self.starting_constant
        w = self.map(noise)

        x = self.initial_block(x, w) # initial block
        
        if current_layer == 0:
            return self.initial_rgb(x)

        for layer in range(current_layer):
            upscaled = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.prog_blocks[layer](upscaled, w)

        upscaled = self.rgb_layers[current_layer - 1](upscaled)
        x = self.rgb_layers[current_layer](x)
        return self.fade_in(upscaled, x, alpha)

    def interpolate(self, upscaled, generated, alpha):
        return generated * alpha + (1 - alpha) * upscaled

    def fade_in(self, upscaled, generated, alpha):
        return torch.tanh(self.interpolate(upscaled, generated, alpha))

class StyleGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride = 1, padding = 1):
        super().__init__()
        self.conv1 = WSConv2d(in_chan, out_chan, kernel_size, stride = stride, padding=padding)
        self.conv2 = WSConv2d(in_chan, out_chan, kernel_size, stride = stride, padding=padding)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x

class StyleGANDiscriminator(nn.Module):
    '''
    Values:
        in_chan: same as the dimension of the constant inputto the generator class
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''
    def __init__(self,
                 in_chan,
                 kernel_size,
                 hidden_chan,
                 img_channels = 3):
        super().__init__()
        
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        
        for _ in range(get_num_steps()): 
            self.prog_blocks.append(
                StyleGANDiscriminatorBlock(hidden_chan, hidden_chan, kernel_size)
            )
            # for the rgb layer, we are doing the reverse of the genrator 
            # (taking an image <img_channels> channels, and passing it through 
            # the Conv2d to get an output with <hidden_chan> channels)
            self.rgb_layers.append(
                WSConv2d(img_channels, hidden_chan, kernel_size=1, stride=1, padding=0)
            )
        
        # rgb layer for the 4x4 input size
        self.final_rgb = WSConv2d(img_channels, in_chan, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.final_rgb)
        # down sampling using avg pool
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  
        # Can increase size (see GitHub for reference)
        self.final_block = nn.Sequential(
            WSConv2d(in_chan + 1, in_chan, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_chan, in_chan, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_chan, 1, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x, alpha, current_layer):
        # where we should start in the list of prog_blocks 
        # (current_layer = 0 means we are at the 4x4 block. = 4 means we are at the 64x64 block)
        cur_step = len(self.prog_blocks) - current_layer

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))
        if current_layer == 0:
            out = self.minibatch_std(out)
            # print(out.shape)
            return self.final_block(out).view(out.shape[0], -1)
        # the following two images are interpolated during training
        # downscaled image using average pooling
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        # output image from the block (current one that we are training)
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        # interpolation
        out = self.fade_in(downscaled, out, alpha)

        # for the rest of the blocks, pass in the image and downscale it, and repeat
        for layer in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[layer](out)
            out = self.avg_pool(out)
        # minibatch std
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def fade_in(self, downscaled, generated, alpha):
        return generated * alpha + (1 - alpha) * downscaled

class StyleGAN(pl.LightningModule): # TODO: Add val loop, spectral norm
    def __init__(self, z_dim, map_hidden_dim, w_dim, in_chan, kernel_size, hidden_chan, lr, b1, b2, c_lambda, image_channels = 3, n_val = 8):
        super().__init__()
        self.save_hyperparameters()
        self.generator = StyleGANGenerator(z_dim, map_hidden_dim, w_dim, in_chan, kernel_size, hidden_chan, image_channels)
        self.critic = StyleGANDiscriminator(in_chan, kernel_size, hidden_chan)
        self.alpha = 0
        self.current_layer = 0
        self.n_val = n_val
        self.c_lambda = c_lambda
        self.gradient_penalty = True
        self.show_generated_images = True
        self.crit_repeats = 3
        self.counter = 0
        self.validation_z = torch.randn(self.hparams.n_val, self.hparams.hidden_chan)
    
    def forward(self, z):
        return self.generator(z, self.alpha, self.current_layer)
    
    def get_crit_loss(self, fake_pred, real_pred, gp, c_lambda):
        return - real_pred.mean() + fake_pred.mean() + c_lambda * gp

    def get_gen_loss(self, crit_fake_pred):
        return - crit_fake_pred.float().mean()
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        z = torch.randn(batch.size(0), self.hparams.z_dim).type_as(batch)
        if optimizer_idx < self.crit_repeats: # train discriminator
            fake = self(z)
            crit_fake_pred = self.critic(fake.detach(), self.alpha, self.current_layer)
            crit_real_pred = self.critic(batch, self.alpha, self.current_layer)
            # gradient penalty
            if self.gradient_penalty:
                epsilon = torch.rand(len(batch), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(self.critic, batch, fake.detach(), epsilon, self.alpha, self.current_layer)
                gp = gradient_penalty(gradient)
                crit_loss = self.get_crit_loss(crit_fake_pred, crit_real_pred, gp, self.c_lambda)
            else:
                crit_loss = self.get_crit_loss(crit_fake_pred, crit_real_pred, 0, 0)
            
            if optimizer_idx == self.crit_repeats - 1: # log last critic loss
                self.log('crit_loss', crit_loss)
            
            return crit_loss
        
        else: # train generator
            fake = self(z)
            crit_fake_pred = self.critic(fake, self.alpha, self.current_layer)
            gen_loss = self.get_gen_loss(crit_fake_pred)
            if self.show_generated_images:
                show_tensor_images(fake.detach(), 8, name = f'images/wgan_gp_small/{self.counter}.png')
                self.counter += 1
            self.log('gen_loss', gen_loss)
            return gen_loss   

    def configure_optimizers(self): # Add LR Scheduler here
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_gen = optim.Adam([{"params": [param for name, param in self.generator.named_parameters() if "map" not in name]},
                      {"params": self.generator.map.parameters(), "lr": lr * 0.1}], lr=lr, betas=(b1, b2))
        opt_critic = optim.Adam(self.critic.parameters(), lr=4 * lr, betas=(b1, b2))
        
        return ([opt_critic] * self.crit_repeats) + [opt_gen]

    def on_epoch_end(self):
        z = self.validation_z.to(device) # see if this can be changed to type_as
        sample_images = self(z)
        show_tensor_images(sample_images, self.n_val, name = f'images/wgan_gp_small/{self.counter}.png')
        self.counter += 1
        # grid = make_grid(sample_images)
        # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

class TrainCallback(pl.callbacks.Callback):
    def __init__(self, model, datamodule, epochs_before_resolution_update = 2, steps_till_max_alpha = None):
        self.dataset = datamodule
        self.model = model

        # Could use learning scheduler to improve this
        self.epochs_before_resolution_update = epochs_before_resolution_update
        self.steps_till_max_alpha = steps_till_max_alpha
        self.steps_per_epoch = ceil(total_images / BATCH_SIZE)
        if not self.steps_till_max_alpha:
            self.steps_till_max_alpha = self.steps_per_epoch
        self.alpha_step = 1 / self.steps_till_max_alpha
        assert self.epochs_before_resolution_update * self.steps_per_epoch >= self.steps_till_max_alpha, "Increase alpha to 1 before moving on to next resolution"
        self.max_resolution = full_resolution
        self.current_resolution = 4
        self.epoch_layer_count = 0
        self.step_alpha_count = 0
        self.max_layer = get_num_steps()
        self.mean, self.std = self.dataset.get_data_mean_std()

    def on_train_epoch_start(self, trainer, pl_module):
        if self.update_layer():
            self.change_resolution(self.get_resolution())
        print(f'Now training in {self.current_resolution} x {self.current_resolution} resolution')

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_layer_count += 1
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): # called on every batch
        self.update_alpha()

    def update_alpha(self):
        if self.step_alpha_count < self.steps_till_max_alpha:
            self.step_alpha_count += 1
            self.model.alpha = min(round(self.model.alpha + self.alpha_step, 8), 1.0)
            
    def update_layer(self):
        if self.model.current_layer == self.max_layer:
            return False # skip layer updates (already in full resolution)
        
        if self.epoch_layer_count == self.epochs_before_resolution_update:
            self.model.current_layer += 1
            self.epoch_layer_count = 0
            self.step_alpha_count = 0
            self.model.alpha = 0
        return True # updated layer
    
    def get_resolution(self):
        return 4 * 2 ** self.model.current_layer
    
    def change_resolution(self, resolution):
        # Hack to change the resolution during training
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.Normalize(self.mean, self.std)
        ])
        self.dataset.set_transform(transform)
        self.current_resolution = resolution

def disable_progan(model, cond):
    if cond:
        model.alpha = 1
        model.current_layer = 4
        callback = None
    else:
        steps_per_epoch = ceil(total_images / BATCH_SIZE)
        callback = TrainCallback(model, dm, epochs_before_resolution_update=20, steps_till_max_alpha = steps_per_epoch * 16)
        # assert callback.epochs_before_resolution_update * 4 + 1 <= epochs, "Please train to full resolution (64 x 64)!"
    return callback

def disable_gp(model, disable):
    model.gradient_penalty = not disable

def enable_show_images(model, show_images):
    model.show_generated_images = show_images

config = {
    "z_dim": 64,
    "map_hidden_dim": 32,
    "w_dim": 64,
    "in_chan": 64,
    "kernel_size": 3,
    "hidden_chan": 64,
    "lr": 2e-4,
    "b1": 0.5,
    "b2": 0.99,
    "c_lambda": 10
}

model = StyleGAN(**config)
wandb_logger = WandbLogger()
callback = disable_progan(model, False)
disable_gp(model, False)
enable_show_images(model, False)

trainer = pl.Trainer(gpus=AVAIL_GPUS, precision = 16, callbacks=callback, logger=wandb_logger, log_every_n_steps=10, default_root_dir='checkpoints')
trainer.fit(model, dm)

def generate_images(model, n_samples, rows, truncation = 0.7):
    z = get_truncated_noise(n_samples, config['z_dim'], truncation).to(device)
    image_tensor = model(z)
    show_tensor_images(image_tensor, rows)
generate_images(model.to(device), 25, 5)
