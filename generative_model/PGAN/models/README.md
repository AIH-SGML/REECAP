# Models

## Architecture

**networks/**: networks' architectures

**utils/**: various utilities

**datasets/**: specific dataset models

**loss_criterions/**: gan loss criterions. Whether they are "basic" (MSE...) or
                     more model specific (AC-GAN)

**trainer/**: wrappers used to handle the GAN's training. Things like: number of iterations, logging, visualization... will be handled here.

---
**base_gan.py** : the reference structure for GANs. All GANs must inherit from this class.

This mother BaseGANs handles:
* the GAN training sequence as described in Generative Adversarial Nets
* GPU and multi-GPU
* saving and loading into a file
* gradient penalty
* nature of the loss
* conditional generation (ACGAN)

What should be handled in a child class
* nature of the G and D networks
* kind of optimizers used (will be moved in BaseGAN)
* and other functions model specific

**progressive_gan.py**: an implementation of [NVIDIA's progressive gan](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf). This class inherits from the BaseGAN abstract class.

**trainer/std_p_gan_config.py**: standard configuration for a ProgressiveGAN training.