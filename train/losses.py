import torch
import torch.nn.functional as F

REGRESSION_PAIRS = {
    "style_content":   ("style", "content"),
    "style_context":   ("style", "context"),
    "content_context": ("content", "context"),
    "context_content": ("context", "content"),
}
CLASSIFICATION_SOURCES = ("content", "context")


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def _normed_codes(model, human_embed, agent_embed):
    style_h, content_h, context_h = model.encode(human_embed, sample=False)
    style_a, content_a, context_a = model.encode(agent_embed, sample=False)

    codes = {
        "style":   torch.cat([style_h, style_a], dim=0),
        "content": torch.cat([content_h, content_a], dim=0),
        "context": torch.cat([context_h, context_a], dim=0),
    }
    # Non-affine LayerNorm: removes the scale/shift trick
    codes = {k: F.layer_norm(v, v.shape[-1:]) for k, v in codes.items()}

    n = human_embed.size(0)
    labels = torch.cat([
        torch.zeros(n, device=human_embed.device),
        torch.ones(n, device=human_embed.device),
    ]).unsqueeze(-1)
    return codes, labels


def probe_loss(model, human_embed, agent_embed):
    """Trains the PROBES only. Codes are detached."""
    with torch.no_grad():
        codes, labels = _normed_codes(model, human_embed, agent_embed)

    losses = {}
    for name, (src, tgt) in REGRESSION_PAIRS.items():
        pred = model.probes[name](codes[src])
        losses[name] = F.mse_loss(pred, codes[tgt])

    for src in CLASSIFICATION_SOURCES:
        name = f"{src}_style"
        logits = model.probes[name](codes[src])
        losses[name] = F.binary_cross_entropy_with_logits(logits, labels)

    return {"total_loss": sum(losses.values()), **losses}


def confusion_loss(model, human_embed, agent_embed):
    """Trains the ENCODERS only. Call with probes frozen."""
    codes, _ = _normed_codes(model, human_embed, agent_embed)

    losses = {}
    for name, (src, tgt) in REGRESSION_PAIRS.items():
        pred = model.probes[name](codes[src])
        mean_target = codes[tgt].detach().mean(dim=0, keepdim=True)
        losses[name] = F.mse_loss(pred, mean_target.expand_as(pred))

    for src in CLASSIFICATION_SOURCES:
        name = f"{src}_style"
        logits = model.probes[name](codes[src])
        losses[name] = F.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, 0.5)
        )

    return {"total_loss": sum(losses.values()), **losses}


def pair_align_loss(a, b):
    a = F.layer_norm(a, a.shape[-1:])
    b = F.layer_norm(b, b.shape[-1:])

    return F.mse_loss(a, b)


def vae_loss(x, recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_per_example = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    kl_loss = kl_per_example.mean()
    total_loss = recon_loss + beta * kl_loss

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }


def compute_style_loss(model, style_embed, style_labels):
    style_logits    = model.style_classifier(style_embed).squeeze(-1)

    return F.binary_cross_entropy_with_logits(
        style_logits,
        style_labels.float().reshape(-1)
    )


def compute_content_loss(model, human_embed, agent_embed):
    cfg = model.cfg

    logits_h        = F.mse_loss(model.content_regressor(human_embed), agent_embed.detach())
    logits_a        = F.mse_loss(model.content_regressor(agent_embed), human_embed.detach())
    regression_loss = (logits_h + logits_a) / 2
    retrieval_loss  = pair_retrieval_loss(human_embed, agent_embed)
    total_loss      = regression_loss + cfg.w_retrieval * retrieval_loss

    return {
        "total_loss": total_loss,
        "regression_loss": regression_loss,
        "retrieval_loss": retrieval_loss
    }


def compute_context_loss(model, context_embed, question_embed):
    cfg = model.cfg

    context_logits  = model.context_regressor(context_embed)
    regression_loss = F.mse_loss(context_logits, question_embed)
    retrieval_loss  = pair_retrieval_loss(context_logits, question_embed)
    total_loss      = regression_loss + cfg.w_retrieval * retrieval_loss

    return {
        "total_loss": total_loss,
        "regression_loss": regression_loss,
        "retrieval_loss": retrieval_loss
    }


def pair_retrieval_loss(embed_a, embed_b, temperature=0.1):
    if embed_a.size(0) != embed_b.size(0):
        raise ValueError("Paired batches must have equal sizes.")

    if embed_a.size(0) < 2:
        raise ValueError("Contrastive retrieval needs at least 2 pairs per batch.")

    a = F.normalize(embed_a, p=2, dim=-1)
    b = F.normalize(embed_b, p=2, dim=-1)

    # Similarity of every candidate pair.
    logits  = (a @ b.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    # A --> B and B --> A
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)

    return (loss_ab + loss_ba) / 2


def kl_divergence(mu, logvar):
    return -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=-1).mean()


def reconstruction_loss(model, human_embed, agent_embed, question_embed):
    cfg = model.cfg
    batch_size = human_embed.size(0)
    beta_kl = getattr(cfg, "beta_kl", 0.01)

    # --------------------------------------------------
    # 1. Individual VAE reconstruction losses
    # --------------------------------------------------
    # Style VAE reconstruct human and AI responses.
    # style_h = vae_loss(human_embed, *model.style_encoder(human_embed)[:3], beta=beta_kl)
    # style_a = vae_loss(agent_embed, *model.style_encoder(agent_embed)[:3], beta=beta_kl)
    # style_vae_loss = (style_h["total_loss"] + style_a["total_loss"]) / 2

    # # Content VAE: reconstruct response embeddings.
    # content_h = vae_loss(human_embed, *model.content_encoder(human_embed)[:3], beta=beta_kl)
    # content_a = vae_loss(agent_embed, *model.content_encoder(agent_embed)[:3], beta=beta_kl)
    # content_vae_loss = (content_h["total_loss"] + content_a["total_loss"]) / 2

    # # Context VAE: reconstruct question embeddings.
    # context_q = vae_loss(question_embed, *model.context_encoder(question_embed)[:3], beta=beta_kl)
    # context_vae_loss = context_q["total_loss"]

    # vae_total   = context_vae_loss
    # vae_total = (content_vae_loss + context_vae_loss) / 2
    # # vae_total = (style_vae_loss + content_vae_loss + context_vae_loss) / 3

    # --------------------------------------------------
    # 2. Latent codes for disentanglement objectives
    # --------------------------------------------------

    style_h, content_h, context_h = model.encode(human_embed)
    style_a, content_a, context_a = model.encode(agent_embed)

    human_labels = torch.zeros(batch_size, device=human_embed.device)
    agent_labels = torch.ones(batch_size, device=agent_embed.device)

    style_loss = (
        compute_style_loss(model, style_h, human_labels) + 
        compute_style_loss(model, style_a, agent_labels)
    ) / 2

    content_loss = compute_content_loss(model, content_h, content_a)["total_loss"]

    # Context should encode the question, so encode the question
    # directly rather than treating the response as the question.
    # _, _, context_question = model.encode(question_embed)
    h_context_loss = compute_context_loss(model, context_h, question_embed)["total_loss"]
    a_context_loss = compute_context_loss(model, context_a, question_embed)["total_loss"]
    context_loss   = (h_context_loss + a_context_loss) / 2

    # --------------------------------------------------
    # 3. Combined-generator reconstruction
    # --------------------------------------------------

    human_recon = model.generate(style_h, content_h, context_h)
    agent_recon = model.generate(style_a, content_a, context_a)

    reconstruct_loss = 0.5 * (
        F.mse_loss(human_recon, human_embed) + 
        F.mse_loss(agent_recon, agent_embed)
    )

    # --------------------------------------------------
    # 4. Combine objectives
    # --------------------------------------------------

    total_loss = (
        # vae_total + 
        cfg.w_style * style_loss + 
        cfg.w_content * content_loss + 
        cfg.w_context * context_loss + 
        cfg.w_recon * reconstruct_loss
    )

    return {
        "total_loss": total_loss,
        # "vae_loss": vae_total,
        # "style_vae_loss": style_vae_loss,
        # "content_vae_loss": content_vae_loss,
        # "context_vae_loss": context_vae_loss,
        "style_loss": style_loss,
        "content_loss": content_loss,
        "context_loss": context_loss,
        "reconstruct_loss": reconstruct_loss,
    }


def generation_loss(model, human_embed, agent_embed, question_embed):
    cfg = model.cfg

    style_human, content_human, context_human = model.encode(human_embed)
    style_agent, content_agent, context_agent = model.encode(agent_embed)

    batch_size = human_embed.size(0)

    human_style_label = torch.zeros(batch_size, device=human_embed.device)
    agent_style_label = torch.ones(batch_size, device=agent_embed.device)

    hah_embed = model.generate(style_human, content_agent, context_human)
    aha_embed = model.generate(style_agent, content_human, context_agent)
    haa_embed = model.generate(style_human, content_agent, context_agent)
    ahh_embed = model.generate(style_agent, content_human, context_human)

    hah_style, hah_content, hah_context = model.encode(hah_embed)
    aha_style, aha_content, aha_context = model.encode(aha_embed)
    haa_style, haa_content, haa_context = model.encode(haa_embed)
    ahh_style, ahh_content, ahh_context = model.encode(ahh_embed)

    # Style classification
    style_loss = (
        compute_style_loss(model, hah_style, human_style_label) +
        compute_style_loss(model, aha_style, agent_style_label) +
        compute_style_loss(model, haa_style, human_style_label) +
        compute_style_loss(model, ahh_style, agent_style_label)
    ) / 4

    # Content consistency
    ca, ch = content_agent.detach(), content_human.detach()
    content_loss = (
        F.mse_loss(hah_content, ca) +
        F.mse_loss(aha_content, ch) +
        F.mse_loss(haa_content, ca) +
        F.mse_loss(ahh_content, ch)
    ) / 4
    # content_loss = (
    #     compute_content_loss(model, hah_content, ca)["total_loss"] +
    #     compute_content_loss(model, aha_content, ch)["total_loss"] +
    #     compute_content_loss(model, haa_content, ca)["total_loss"] +
    #     compute_content_loss(model, ahh_content, ch)["total_loss"]
    # ) / 4

    # Context reconstruction
    context_loss = (
        compute_context_loss(model, hah_context, question_embed)["total_loss"] +
        compute_context_loss(model, aha_context, question_embed)["total_loss"] +
        compute_context_loss(model, haa_context, question_embed)["total_loss"] +
        compute_context_loss(model, ahh_context, question_embed)["total_loss"]
    ) / 4

    # Combine objectives
    total_loss  = (
        cfg.w_style * style_loss + 
        cfg.w_content * content_loss + 
        cfg.w_context * context_loss
    )

    return {
        "total_loss": total_loss,
        "style_loss": style_loss,
        "content_loss": content_loss,
        "context_loss": context_loss
    }