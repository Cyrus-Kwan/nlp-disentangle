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
    style_h, content_h, context_h = model.encode(human_embed)
    style_a, content_a, context_a = model.encode(agent_embed)

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


def compute_style_loss(model, style_embed, style_labels):
    style_logits    = model.style_classifier(style_embed).squeeze(-1)

    return F.binary_cross_entropy_with_logits(
        style_logits,
        style_labels.float().reshape(-1)
    )


def compute_content_loss(model, human_embed, agent_embed):
    regression_loss = F.mse_loss(
        model.content_regressor(human_embed), agent_embed.detach()
    )
    retrieval_loss  = pair_retrieval_loss(human_embed, agent_embed)
    total_loss      = regression_loss + retrieval_loss

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


def reconstruction_loss(model, human_embed, agent_embed, question_embed):
    cfg = model.cfg

    style_human, content_human, context_human = model.encode(human_embed)
    style_agent, content_agent, context_agent = model.encode(agent_embed)

    batch_size = human_embed.size(0)

    human_style_label = torch.zeros(batch_size, device=human_embed.device)
    agent_style_label = torch.ones(batch_size, device=agent_embed.device)

    # Style classification
    human_style_loss    = compute_style_loss(model, style_human, human_style_label)
    agent_style_loss    = compute_style_loss(model, style_agent, agent_style_label)
    style_loss          = (human_style_loss + agent_style_loss) / 2

    # Content consistency
    content_loss        = compute_content_loss(model, content_human, content_agent)["total_loss"]

    # Context reconstruction
    human_context_loss  = compute_context_loss(model, context_human, question_embed)["total_loss"]
    agent_context_loss  = compute_context_loss(model, context_agent, question_embed)["total_loss"]
    context_loss        = (human_context_loss + agent_context_loss) / 2

    # Pair align loss
    align_loss = (
        pair_align_loss(content_human, content_agent) + 
        pair_align_loss(context_human, context_agent)
    )
    

    # Reconstruct the original response embeddings
    human_reconstruct, *_   = model.reconstruct(human_embed)
    agent_reconstruct, *_   = model.reconstruct(agent_embed)
    reconstruct_loss        = 0.5 * (
        F.mse_loss(human_reconstruct, human_embed) +
        F.mse_loss(agent_reconstruct, agent_embed)
    )

    # Combine objectives
    total_loss  = (
        cfg.w_style * style_loss + 
        cfg.w_content * content_loss + 
        cfg.w_context * context_loss + 
        cfg.w_recon * reconstruct_loss +
        cfg.w_align * align_loss
    )

    return {
        "total_loss": total_loss,
        "style_loss": style_loss,
        "content_loss": content_loss,
        "context_loss": context_loss,
        "reconstruct_loss": reconstruct_loss
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
    hah_style_loss = compute_style_loss(model, hah_style, human_style_label)
    aha_style_loss = compute_style_loss(model, aha_style, agent_style_label)
    haa_style_loss = compute_style_loss(model, haa_style, human_style_label)
    ahh_style_loss = compute_style_loss(model, ahh_style, agent_style_label)
    style_loss = (hah_style_loss + aha_style_loss + haa_style_loss + ahh_style_loss) / 4

    # Content consistency
    ca, ch = content_agent.detach(), content_human.detach()
    content_loss = (
        F.mse_loss(hah_content, ca) +
        F.mse_loss(aha_content, ch) +
        F.mse_loss(haa_content, ca) +
        F.mse_loss(ahh_content, ch)
    ) / 4

    # Context reconstruction
    hah_context_loss = compute_context_loss(model, hah_context, question_embed)["total_loss"]
    aha_context_loss = compute_context_loss(model, aha_context, question_embed)["total_loss"]
    haa_context_loss = compute_context_loss(model, haa_context, question_embed)["total_loss"]
    ahh_context_loss = compute_context_loss(model, ahh_context, question_embed)["total_loss"]
    context_loss = (hah_context_loss + aha_context_loss + haa_context_loss + ahh_context_loss) / 4

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