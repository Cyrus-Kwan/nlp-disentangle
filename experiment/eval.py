import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

@torch.no_grad()
def evaluate_encoders(model, val_loader, device):
    model.eval()

    all_questions = []
    all_human_style = []
    all_agent_style = []
    all_human_content = []
    all_agent_content = []
    all_human_context = []
    all_agent_context = []

    for batch in val_loader:
        question = batch["question"].to(device)
        human = batch["human"].to(device)
        agent = batch["agent"].to(device)

        # Encode the paired human and AI responses
        style_h, content_h, context_h = model.encode(human)
        style_a, content_a, context_a = model.encode(agent)

        all_questions.append(question.cpu())
        all_human_style.append(style_h.cpu())
        all_agent_style.append(style_a.cpu())
        all_human_content.append(content_h.cpu())
        all_agent_content.append(content_a.cpu())
        all_human_context.append(context_h.cpu())
        all_agent_context.append(context_a.cpu())

    # Combine batches so retrieval is evaluated across the full
    # validation set, rather than only within individual batches.
    questions = torch.cat(all_questions)
    style_h = torch.cat(all_human_style)
    style_a = torch.cat(all_agent_style)
    content_h = torch.cat(all_human_content)
    content_a = torch.cat(all_agent_content)
    context_h = torch.cat(all_human_context)
    context_a = torch.cat(all_agent_context)

    # ---------------------------------------------------------
    # 1. STYLE: human vs. AI classification accuracy
    # ---------------------------------------------------------
    style_embeddings = torch.cat([style_h, style_a], dim=0)

    style_labels = torch.cat([
        torch.zeros(len(style_h)),
        torch.ones(len(style_a)),
    ]).long()

    style_logits = model.style_classifier(
        style_embeddings.to(device)
    ).squeeze(-1).cpu()

    style_predictions = (style_logits >= 0).long()
    style_accuracy = (
        style_predictions == style_labels
    ).float().mean().item()

    # ---------------------------------------------------------
    # 2. CONTENT: paired-response retrieval accuracy
    # ---------------------------------------------------------
    # Each human response should match its paired AI response.
    # Normalize vectors so matrix multiplication gives cosine
    # similarity.
    content_h = F.normalize(content_h, p=2, dim=-1)
    content_a = F.normalize(content_a, p=2, dim=-1)

    content_similarity = content_h @ content_a.T
    targets = torch.arange(len(content_h))

    # Human -> AI and AI -> human retrieval
    h_to_a = (
        content_similarity.argmax(dim=1) == targets
    ).float().mean().item()

    a_to_h = (
        content_similarity.argmax(dim=0) == targets
    ).float().mean().item()

    content_accuracy = (h_to_a + a_to_h) / 2

    # Also report paired cosine similarity for interpretability.
    content_paired_cosine = (
        content_h * content_a
    ).sum(dim=-1).mean().item()

    # ---------------------------------------------------------
    # 3. CONTEXT: question embedding prediction
    # ---------------------------------------------------------
    # The context regressor predicts the original SBERT
    # question embedding from each context representation.
    predicted_context_h = model.context_regressor(
        context_h.to(device)
    ).cpu()

    predicted_context_a = model.context_regressor(
        context_a.to(device)
    ).cpu()

    predicted_context = torch.cat(
        [predicted_context_h, predicted_context_a], dim=0
    )

    # Each question is a candidate target for retrieval.
    # The question embedding is shared by its human and AI pair.
    question_candidates = F.normalize(questions, p=2, dim=-1)
    predicted_context = F.normalize(
        predicted_context, p=2, dim=-1
    )

    # Repeat question indices to align with human and AI contexts.
    context_targets = torch.cat([
        torch.arange(len(questions)),
        torch.arange(len(questions)),
    ])

    context_similarity = predicted_context @ question_candidates.T

    context_retrieval_accuracy = (
        context_similarity.argmax(dim=1) == context_targets
    ).float().mean().item()

    # Cosine similarity with the correct question for each response
    true_questions = questions.repeat(2, 1)
    context_cosine = F.cosine_similarity(
        predicted_context, true_questions, dim=-1
    ).mean().item()

    # ---------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------
    results = pd.DataFrame([
        {
            "Encoder": "Style",
            "Metric": "Human/AI classification accuracy",
            "Score": style_accuracy,
        },
        {
            "Encoder": "Content",
            "Metric": "Paired retrieval accuracy (top-1)",
            "Score": content_accuracy,
        },
        {
            "Encoder": "Content",
            "Metric": "Paired cosine similarity",
            "Score": content_paired_cosine,
        },
        {
            "Encoder": "Context",
            "Metric": "Question retrieval accuracy (top-1)",
            "Score": context_retrieval_accuracy,
        },
        {
            "Encoder": "Context",
            "Metric": "Question cosine similarity",
            "Score": context_cosine,
        },
    ])

    results["Score"] = results["Score"].round(4)

    return results

def collect_dataset(data_loader):
    all_questions   = []
    all_human       = []
    all_agent       = []

    for batch in data_loader:
        all_questions.append(batch["question"])
        all_human.append(batch["human"])
        all_agent.append(batch["agent"])

    return {
        "question": torch.cat(all_questions),
        "human": torch.cat(all_human),
        "agent": torch.cat(all_agent)
    }

@torch.no_grad()
def encode_dataset(model, data_loader, device):
    model.eval()

    dataset = collect_dataset(data_loader)

    # Encode the paired human and AI responses
    style_h, content_h, context_h = model.encode(dataset["human"].to(device))
    style_a, content_a, context_a = model.encode(dataset["agent"].to(device))

    dataset["Style"]    = (style_h.cpu(), style_a.cpu())
    dataset["Content"]  = (content_h.cpu(), content_a.cpu())
    dataset["Context"]  = (context_h.cpu(), context_a.cpu())

    return dataset

def top1_retrieval(query_embed, candidate_embed):
    # Row i of the queries should be closest to row i of the candidates
    query_embed     = F.normalize(query_embed, p=2, dim=-1)
    candidate_embed = F.normalize(candidate_embed, p=2, dim=-1)

    similarity  = query_embed @ candidate_embed.T
    targets     = torch.arange(len(query_embed))

    return (
        similarity.argmax(dim=1) == targets
    ).float().mean().item()

def paired_retrieval(human_embed, agent_embed):
    # Human -> AI and AI -> human retrieval, same as evaluate_encoders
    h_to_a = top1_retrieval(human_embed, agent_embed)
    a_to_h = top1_retrieval(agent_embed, human_embed)

    return (h_to_a + a_to_h) / 2

def probe_style(train_h, train_a, test_h, test_a):
    # A fresh linear probe, so the score does not depend on
    # the classifier that was trained together with the model.
    train_x = torch.cat([train_h, train_a]).numpy()
    test_x  = torch.cat([test_h, test_a]).numpy()

    train_y = np.r_[np.zeros(len(train_h)), np.ones(len(train_a))]
    test_y  = np.r_[np.zeros(len(test_h)), np.ones(len(test_a))]

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000)
    )
    probe.fit(train_x, train_y)

    return probe.score(test_x, test_y)

def probe_question(train_h, train_a, train_q, test_h, test_a, test_q):
    # A fresh linear map from any embedding into the question
    # space, then retrieve the correct question for each response.
    train_x = torch.cat([train_h, train_a]).numpy()
    test_x  = torch.cat([test_h, test_a]).numpy()
    train_y = torch.cat([train_q, train_q]).numpy()

    probe = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    probe.fit(train_x, train_y)

    predicted   = torch.tensor(probe.predict(test_x), dtype=torch.float32)

    # Rank each prediction against the test questions only
    predicted   = F.normalize(predicted, p=2, dim=-1)
    questions   = F.normalize(test_q, p=2, dim=-1)
    targets     = torch.arange(len(test_q)).repeat(2)

    return (
        (predicted @ questions.T).argmax(dim=1) == targets
    ).float().mean().item()

def probe_matrix(train_set, test_set, encoders=("Style", "Content", "Context")):
    # Diagonal: each encoder should hold its own information.
    # Off-diagonal: everything else should be close to chance.
    rows = []

    for encoder in encoders:
        train_h, train_a    = train_set[encoder]
        test_h, test_a      = test_set[encoder]

        style_score = probe_style(
            train_h, train_a,
            test_h, test_a
        )
        pair_score = paired_retrieval(test_h, test_a)
        question_score = probe_question(
            train_h, train_a, train_set["question"],
            test_h, test_a, test_set["question"]
        )

        rows.append({"Encoder": encoder, "Metric": "Human/AI probe accuracy", "Score": style_score})
        rows.append({"Encoder": encoder, "Metric": "Paired retrieval accuracy (top-1)", "Score": pair_score})
        rows.append({"Encoder": encoder, "Metric": "Question probe retrieval (top-1)", "Score": question_score})

    return pd.DataFrame(rows)

def collapse_check(test_set, encoders=("Style", "Content", "Context")):
    # A collapsed encoder makes every vector look alike, so true
    # pairs are no closer than random pairs and few dimensions are used.
    rows = []

    for encoder in encoders:
        human_embed, agent_embed = test_set[encoder]

        human_unit = F.normalize(human_embed, p=2, dim=-1)
        agent_unit = F.normalize(agent_embed, p=2, dim=-1)

        paired_cosine = (human_unit * agent_unit).sum(dim=-1).mean().item()
        random_cosine = (human_unit * agent_unit.roll(1, dims=0)).sum(dim=-1).mean().item()

        # Effective dimensions: how many directions carry the variance
        all_embeds      = torch.cat([human_embed, agent_embed])
        eigenvalues     = torch.linalg.eigvalsh(torch.cov(all_embeds.T)).clamp(min=0)
        effective_dims  = (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()

        rows.append({"Encoder": encoder, "Metric": "Paired cosine similarity", "Score": paired_cosine})
        rows.append({"Encoder": encoder, "Metric": "Random-pair cosine similarity", "Score": random_cosine})
        rows.append({"Encoder": encoder, "Metric": "Effective dimensions", "Score": effective_dims})

    return pd.DataFrame(rows)

@torch.no_grad()
def swap_test(model, test_set, device):
    model.eval()

    human = test_set["human"].to(device)
    agent = test_set["agent"].to(device)

    style_h, style_a        = (embed.to(device) for embed in test_set["Style"])
    content_h, content_a    = (embed.to(device) for embed in test_set["Content"])
    context_h, context_a    = (embed.to(device) for embed in test_set["Context"])

    # ---------------------------------------------------------
    # 1. Self-reconstruction
    # ---------------------------------------------------------
    recon_h, *_ = model.reconstruct(human)
    recon_a, *_ = model.reconstruct(agent)

    recon_cosine = 0.5 * (
        F.cosine_similarity(recon_h, human, dim=-1).mean().item() +
        F.cosine_similarity(recon_a, agent, dim=-1).mean().item()
    )

    # ---------------------------------------------------------
    # 2. Cross-generation
    # ---------------------------------------------------------
    # The paraphrase shares the human response's content, so human
    # style + AI content should rebuild the human response.
    hah_embed = model.generate(style_h, content_a, context_h)
    aha_embed = model.generate(style_a, content_h, context_a)

    swap_retrieval = 0.5 * (
        top1_retrieval(hah_embed.cpu(), human.cpu()) +
        top1_retrieval(aha_embed.cpu(), agent.cpu())
    )

    # The style classifier should follow the style donor
    hah_style, hah_content, _ = model.encode(hah_embed)
    aha_style, aha_content, _ = model.encode(aha_embed)

    hah_human = (model.style_classifier(hah_style).squeeze(-1) < 0).float().mean().item()
    aha_agent = (model.style_classifier(aha_style).squeeze(-1) >= 0).float().mean().item()

    style_follow = (hah_human + aha_agent) / 2

    # The content should still point back to the content donor
    content_follow = 0.5 * (
        top1_retrieval(hah_content.cpu(), content_a.cpu()) +
        top1_retrieval(aha_content.cpu(), content_h.cpu())
    )

    return pd.DataFrame([
        {"Encoder": "Generator", "Metric": "Self-reconstruction cosine similarity", "Score": recon_cosine},
        {"Encoder": "Generator", "Metric": "Swap rebuilds the right response (top-1)", "Score": swap_retrieval},
        {"Encoder": "Generator", "Metric": "Swap style follows the style donor", "Score": style_follow},
        {"Encoder": "Generator", "Metric": "Swap content follows the content donor (top-1)", "Score": content_follow},
    ])

def evaluate_disentangler(model, train_loader, test_loader, device):
    # Probes are fitted on the training set and scored on the test set
    train_set = encode_dataset(model, train_loader, device)
    test_set  = encode_dataset(model, test_loader, device)

    results = pd.concat([
        probe_matrix(train_set, test_set),
        collapse_check(test_set),
        swap_test(model, test_set, device)
    ], ignore_index=True)

    results["Score"] = results["Score"].round(4)

    return results

def evaluate_input(train_loader, test_loader):
    # Baseline: the same checks on the raw SBERT vectors, with no model
    train_set = collect_dataset(train_loader)
    test_set  = collect_dataset(test_loader)

    train_set["Input"]  = (train_set["human"], train_set["agent"])
    test_set["Input"]   = (test_set["human"], test_set["agent"])

    results = pd.concat([
        probe_matrix(train_set, test_set, encoders=("Input",)),
        collapse_check(test_set, encoders=("Input",))
    ], ignore_index=True)

    results["Score"] = results["Score"].round(4)

    return results