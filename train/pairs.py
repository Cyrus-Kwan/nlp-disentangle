import torch
from torch.utils.data import Dataset, DataLoader

class SummaryPairDataset(Dataset):
    def __init__(self, question_embed, human_embed, agent_embed):
        assert len(question_embed) == len(human_embed)
        assert len(question_embed) == len(agent_embed)

        self.question_embed = question_embed.float()
        self.human_embed    = human_embed.float()
        self.agent_embed    = agent_embed.float()

    def __len__(self):
        return len(self.question_embed)

    def __getitem__(self, idx):
        return {
            "question": self.question_embed[idx],
            "human": self.human_embed[idx],
            "agent": self.agent_embed[idx]
        }

def make_generated_embeds(model, human_embed, agent_embed):
    style_human, content_human, context_human = model.encode(human_embed)
    style_agent, content_agent, context_agent = model.encode(agent_embed)

    # Self-reconstruction
    recon_human = model.generate(style_human, content_human, context_human)
    recon_agent = model.generate(style_agent, content_agent, context_agent)

    # Cross-generation
    hah_embed   = model.generate(style_human, content_agent, context_human)
    aha_embed   = model.generate(style_agent, content_human, context_agent)
    haa_embed   = model.generate(style_human, content_agent, context_agent)
    ahh_embed   = model.generate(style_agent, content_human, context_human)

    return torch.cat(
        [recon_human, recon_agent, hah_embed, aha_embed, haa_embed, ahh_embed], 
        dim=0
    )