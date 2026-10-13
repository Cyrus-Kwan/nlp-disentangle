import torch
import torch.nn.functional as F

from train.pairs import *
from train.losses import *

def train_step(model, batch, gene_optim, disc_optim):
    question_embed  = batch["question"]
    human_embed     = batch["human"]
    agent_embed     = batch["agent"]

    '''Update discriminator'''
    model.train()
    disc_optim.zero_grad(set_to_none=True)

    # Do not let dscriminator training update the encoders or generator
    with torch.no_grad():
        pair_embeds = make_generated_embeds(model, human_embed, agent_embed)
    
    true_embeds = torch.cat([human_embed, agent_embed], dim=0)

    pair_logits = model.discriminator(pair_embeds)
    true_logits = model.discriminator(true_embeds)

    pair_labels = torch.zeros_like(pair_logits)
    true_labels = torch.ones_like(true_logits)

    pair_loss   = F.binary_cross_entropy_with_logits(pair_logits, pair_labels)
    true_loss   = F.binary_cross_entropy_with_logits(true_logits, true_labels)

    disc_loss   = (pair_loss + true_loss) / 2
    disc_loss.backward()
    disc_optim.step()

    '''Update encoders, generator, and auxiliary heads'''
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)

    gene_optim.zero_grad(set_to_none=True)

    self_loss   = reconstruction_loss(model, human_embed, agent_embed, question_embed)
    cross_loss  = generation_loss(model, human_embed, agent_embed, question_embed)

    # Recompute with gradients enabled for generator training
    gene_embeds = make_generated_embeds(model, human_embed, agent_embed)
    gene_logits = model.discriminator(gene_embeds)
    disc_labels = torch.ones_like(gene_logits)
    
    adv_loss    = F.binary_cross_entropy_with_logits(gene_logits, disc_labels)
    gene_loss   = (
        self_loss["total_loss"] +
        cross_loss["total_loss"] + 
        adv_loss 
    )

    gene_loss.backward()
    gene_optim.step()

    # Re-enable discriminator gradients for the next step
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(True)

    return{
        "generator_loss": gene_loss.detach().item(), 
        "self_loss": self_loss["total_loss"].detach().item(),
        "cross_loss": cross_loss["total_loss"].detach().item(),
        "adversarial_loss": adv_loss.detach().item(),
        "discriminator_loss": disc_loss.detach().item()
    }

class AlternatingState:
    def __init__(
        self,
        d_acc_low=0.65,
        d_acc_high=0.80,
        ema_alpha=0.1,
        max_consecutive_G=3,
    ):
        self.d_acc_low = d_acc_low
        self.d_acc_high = d_acc_high
        self.ema_alpha = ema_alpha
        self.max_consecutive_G = max_consecutive_G

        self.ema_d_acc = None
        self.last_update = "G"
        self.consecutive_G = 0

        self.n_gen = 0
        self.n_dis = 0

    def should_update_discriminator(self):
        # Bootstrap: always train D at the beginning.
        if self.ema_d_acc is None:
            return True

        # Never let G monopolize training.
        if self.consecutive_G >= self.max_consecutive_G:
            return True

        # D is weak: train D.
        if self.ema_d_acc < self.d_acc_low:
            return True

        # D is strong: favor G, but only up to the cap.
        if self.ema_d_acc > self.d_acc_high:
            return False

        # Intermediate region: alternate.
        return self.last_update == "G"

    def record_dis(self, loss, accuracy):
        self.n_dis += 1
        self.last_update = "D"
        self.consecutive_G = 0

        accuracy = float(accuracy)

        if self.ema_d_acc is None:
            self.ema_d_acc = accuracy
        else:
            a = self.ema_alpha
            self.ema_d_acc = (
                (1 - a) * self.ema_d_acc
                + a * accuracy
            )

    def record_gen(self, loss):
        self.n_gen += 1
        self.last_update = "G"
        self.consecutive_G += 1


def train_iteration(
    model, batch, optim_G, optim_D, state, lambda_cross=1.0, lambda_adv=0.05, grad_clip=5.0,
):
    logs = {}

    question    = batch["question"]
    human       = batch["human"]
    agent       = batch["agent"]

    model.train()

    # ==================================================
    # DISCRIMINATOR UPDATE
    # ==================================================
    if state.should_update_discriminator():
        optim_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            pair_embeds = make_generated_embeds(
                model, human, agent
            )

        true_embeds = torch.cat([human, agent], dim=0)

        true_logits = model.discriminator(true_embeds)
        pair_logits = model.discriminator(pair_embeds)

        true_labels = torch.ones_like(true_logits)
        pair_labels = torch.zeros_like(pair_logits)

        loss_D_real = F.binary_cross_entropy_with_logits(
            true_logits, true_labels
        )
        loss_D_fake = F.binary_cross_entropy_with_logits(
            pair_logits, pair_labels
        )

        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        # Measure D's predictive performance before its update.
        with torch.no_grad():
            true_acc = ((true_logits >= 0).float().mean())
            pair_acc = ((pair_logits < 0).float().mean())
            disc_acc = 0.5 * (true_acc + pair_acc)

        loss_D.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)

        optim_D.step()

        state.record_dis(
            loss_D.detach().item(),
            disc_acc.item(),
        )

        logs.update({
            "updated": "D",
            "loss_D": loss_D.detach().item(),
            "loss_D_real": loss_D_real.detach().item(),
            "loss_D_fake": loss_D_fake.detach().item(),
            "disc_acc": disc_acc.item(),
            "ema_disc_acc": state.ema_d_acc,
        })

    # ==================================================
    # GENERATOR UPDATE
    # ==================================================
    else:
        optim_G.zero_grad(set_to_none=True)

        # Freeze D's parameters, but preserve gradients
        # flowing through D into the generated embeddings.
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(False)

        try:
            # Real-data objectives
            self_parts = reconstruction_loss(model, human, agent, question)

            # Factor recombination objectives
            cross_parts = generation_loss(model, human, agent, question)

            # Adversarial objective: generated embeddings
            # should be classified as real.
            pair_embeds = make_generated_embeds(model, human, agent)

            pair_logits     = model.discriminator(pair_embeds)
            true_targets    = torch.ones_like(pair_logits)

            loss_adv = F.binary_cross_entropy_with_logits(pair_logits, true_targets)

            loss_G = (self_parts["total_loss"] + lambda_cross * cross_parts["total_loss"] + lambda_adv * loss_adv)
            loss_G.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.generator_parameters(), grad_clip)

            optim_G.step()

        finally:
            # Restore D for its next update.
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(True)

        state.record_gen(loss_G.detach().item())

        logs.update({
            "updated": "G",
            "loss_G": loss_G.detach().item(),
            "loss_self": self_parts["total_loss"].detach().item(),
            "loss_cross": cross_parts["total_loss"].detach().item(),
            "loss_adv": loss_adv.detach().item(),
            "style_loss": self_parts["style_loss"].detach().item(),
            "content_loss": self_parts["content_loss"].detach().item(),
            "context_loss": self_parts["context_loss"].detach().item(),
            "reconstruct_loss": (
                self_parts["reconstruct_loss"].detach().item()
            ),
        })

    return logs