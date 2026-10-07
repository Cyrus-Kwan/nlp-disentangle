# NLP Disentanglement
This is an NLP project that aims to separate style, content, and residuals from input text.

### Contribution
- Cyrus Kwan
- Bishal Sarker
- Seth Yap Wing Shen

## Dataset
Raw human responses gathered from the [askreddit dataset](https://huggingface.co/datasets/suyanwen/askreddit_processed).
AI generated responses for two additional features: AI response, and AI paraphrase

## Project Structure
```
nlp-disentangle
│
├── data/
│   ├── askreddit.pkl
│   ├── train_set.pkl
│   └── test_set.pkl
│
├── model/
│   ├── encoders.py
│   ├── discriminator.py
│   ├── generator.py
│   ├── blocks.py
│   └── disentangler.py
│
├── train/
│   ├── losses.py
│   ├── pairs.py
│   ├── step.py
│   └── teacher.py
│
└── experiment/
    └── eval.py
```

## Losses
Twelve losses feed the generative objective $\mathcal{L}_{Gen}$:

| # | Loss | Eq. | What it buys you |
|---|---|---|---|
| 1 | $\mathcal{L}^S_{Recon}$ | (2) | codes are complete enough to rebuild the input |
| 2 | $\mathcal{L}^S_{KL}$ | (4) | `z` stays close to a standard Gaussian, so it's sampleable |
| 3 | $\mathcal{L}^S_{Gen}$ | (8) | output looks like real response text |
| 4 | $\mathcal{L}_{Sty}$ | (10) | `e_s` contains style information |
| 5 | $\mathcal{L}_{Con}$ | (10) | `e_c` contains content information |
| 6 | $\mathcal{L}^S_{Cycle}$ | (9) | the generator cannot ignore `z` |
| 7 | $\mathcal{L}^{CI}_{Gen}$ | (11) | style codes are interchangeable across samples |
| 8 | $\mathcal{L}^{CC}_{Gen}$ | (12) | unseen combinations still look real |
| 9 | $\mathcal{L}^{CC}_{CycleE}$ | (13) | mixed samples retain their input codes |
| 10 | $\mathcal{L}^{CC}_{CycleX}$ | (14) | codes survive a full scatter-and-gather round trip |
| 11 | $\mathcal{L}^{KL}_{Sty}$ | (15) | fake samples classify as the right style |
| 12 | $\mathcal{L}^{KL}_{Con}$ | (15) | fake samples loss is minimized as the right content to human and machine responses|
