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