import torch
import torch.nn.functional as F

def compute_style_loss(model, style_embed, style_labels):
    style_logits    = model.style_classifier(style_embed).squeeze(-1)

    return F.binary_cross_entropy_with_logits(
        style_logits,
        style_labels.float().reshape(-1)
    )

def compute_content_loss(model, human_embed, agent_embed):
    human_logits = model.content_regressor(human_embed)
    agent_logits = model.content_regressor(agent_embed)

    return F.mse_loss(human_logits, agent_logits)

def compute_context_loss(model, context_embed, question_embed):
    context_logits  = model.context_regressor(context_embed)

    return F.mse_loss(context_logits, question_embed)

def reconstruction_loss(model, human_embed, agent_embed, question_embed):
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
    content_loss        = compute_content_loss(model, content_human, content_agent)

    # Context reconstruction
    human_context_loss  = compute_context_loss(model, context_human, question_embed)
    agent_context_loss  = compute_context_loss(model, context_agent, question_embed)
    context_loss        = (human_context_loss + agent_context_loss) / 2

    # Reconstruct the original response embeddings
    human_reconstruct, *_   = model.reconstruct(human_embed)
    agent_reconstruct, *_   = model.reconstruct(agent_embed)
    reconstruct_loss        = 0.5 * (
        F.mse_loss(human_reconstruct, human_embed) +
        F.mse_loss(agent_reconstruct, agent_embed)
    )

    # Combine objectives
    total_loss  = (
        style_loss + content_loss + context_loss + reconstruct_loss
    )

    return {
        "total_loss": total_loss,
        "style_loss": style_loss,
        "content_loss": content_loss,
        "context_loss": context_loss,
        "reconstruct_loss": reconstruct_loss
    }

def generation_loss(model, human_embed, agent_embed, question_embed):
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
    content_loss = (
        F.mse_loss(hah_content, content_agent) +
        F.mse_loss(aha_content, content_human) +
        F.mse_loss(haa_content, content_agent) +
        F.mse_loss(ahh_content, content_human)
    ) / 4

    # Context reconstruction
    hah_context_loss = compute_context_loss(model, hah_context, question_embed)
    aha_context_loss = compute_context_loss(model, aha_context, question_embed)
    haa_context_loss = compute_context_loss(model, haa_context, question_embed)
    ahh_context_loss = compute_context_loss(model, ahh_context, question_embed)
    context_loss = (hah_context_loss + aha_context_loss + haa_context_loss + ahh_context_loss) / 4

    # Combine objectives
    total_loss = (style_loss + content_loss + context_loss)

    return {
        "total_loss": total_loss,
        "style_loss": style_loss,
        "content_loss": content_loss,
        "context_loss": context_loss
    }