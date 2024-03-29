
import torch
from torch.nn import BCEWithLogitsLoss

class GANLoss:
    """ Base class for all losses

        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

class StandardGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2
        #return torch.mean(real_loss) + torch.mean(fake_loss)

    def gen_loss(self, _, fake_samps, height, alpha):
        preds = self.dis(fake_samps, height, alpha)
        #loss = self.criterion(torch.squeeze(preds),
        #                      torch.ones(fake_samps.shape[0]).to(fake_samps.device)) 
        #return torch.mean(loss)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))
        
class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=True):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.dis(merged, height, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, height, alpha)
        real_out = self.dis(real_samps, height, alpha)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps, height, alpha))

        return loss
    
class RaLSGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)
        self.labels = 0.5

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        outputR = self.dis(real_samps, height, alpha)
        outputF = self.dis(fake_samps, height, alpha)
        return 0.5 * torch.mean((outputR - torch.mean(outputF) - self.labels) ** 2) + \
               0.5 * torch.mean((outputF - torch.mean(outputR) + self.labels) ** 2)

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        outputR = self.dis(real_samps, height, alpha)
        outputF = self.dis(fake_samps, height, alpha)
        return 0.5 * torch.mean((outputR - torch.mean(outputF) + self.labels) ** 2) + \
               0.5 * torch.mean((outputF - torch.mean(outputR) - self.labels) ** 2)