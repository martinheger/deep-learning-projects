# Deep Learning Projects

**[English](README.md) | [Česky](README_cz.md)**

Three practical projects from a Deep Learning course: image classification using CNN (with a deployed live app), a custom transformer trained on Czech laws, and fine-tuning Gemma 3 for structured output.

## 1. CNN Image Classifier and Live App

[Try the live demo](https://huggingface.co/spaces/MartinH-01/Flowers)

Convolutional neural network (fastai) classifying images (flowers / Fashion-MNIST). The model is deployed as a publicly available Gradio app on Hugging Face Spaces — upload an image, and the app returns a prediction with class probabilities.

- `app.py` — source code of the Gradio app (same code running on HF Spaces)
- `notebooks/` — training and experiments (Fashion-MNIST baseline, flower classification, final app model)

The model (`model_cleansplit.pkl`, ~50 MB) is not included in the repo — the app runs directly on HF Spaces where the model is hosted.

## 2. Custom Transformer — Tokenization and Generation on Czech Laws

Implementation of a GPT-style transformer (based on [Karpathy's "Let's build GPT" tutorial](https://github.com/karpathy/ng-video-lecture)) trained on a corpus of Czech laws, including experiments with tokenization and model size.

Key findings (see [shrnuti-vysledky.txt](./2-transformer-tokenization/shrnuti-vysledky.txt) for full analysis):

- Tested two levels of tokenization (simple vs. improved), vocabulary of approx. 200k subword tokens
- Achieved metrics: train loss approx. 2.4, val loss approx. 3.1 — the capacity of the simple transformer was practically exhausted on this dataset
- Examples of generated text for empty vs. specific prompts in [example-outputs/](./2-transformer-tokenization/example-outputs/) — the model accurately mimics the style of legal texts, even though the content is "hallucinated"

```text
2-transformer-tokenization/
├── notebooks/            # tokenization (2 variants) + dataset preparation
├── example-outputs/      # samples of generated text
├── merged_zakony.md      # source corpus (merged legal texts)
└── shrnuti-vysledky.txt  # summary of experiments and conclusions
```

The notebook `SimpleTokenizationMergedZakony.ipynb` assumes a Google Colab environment (loading data from `/content/`). To run locally, the path must be adjusted.

## 3. Fine-tuning Gemma 3 — Structured Text Analysis

Fine-tuning the Gemma 3 model (via [unsloth](https://github.com/unslothai/unsloth)) for structured text analysis of song lyrics (alternative title, summary, mood evaluation) — dataset of 200 lyrics by Czech artists (Karel Kryl, Karel Plíhal, Ivan Mládek, Svěrák & Uhlíř).

Key findings (see [shrnuti.txt](./3-gemma-finetune-mood-classifier/shrnuti.txt)):

- The model successfully learned to follow the required output structure
- On 100 training examples, the fine-tuned model did not significantly outperform the baseline model — the small dataset limited improvements
- The model tended to default to a "safe middle" when evaluating mood
- Observed a shift towards English at higher generation temperatures

```text
3-gemma-finetune-mood-classifier/
├── gemma3_finetune_final.ipynb      # fine-tuning pipeline
├── Vyuziti_GPT_api_dataset.ipynb    # dataset preparation via GPT API
├── train_100.jsonl / test_99.jsonl  # training and test data
└── shrnuti.txt                      # summary of results and observations
```

## Tech Stack

Python, PyTorch, fastai, Gradio, Hugging Face Spaces, unsloth, OpenAI API

## Authorship and Context

Group assignments for the course PřF:M7DataSP – Advanced Data Science Practicum (2025) at Masaryk University, based on the course repository [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) and the shared team repository [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team).

In collaboration with Lucia Kajanová and Eva Kopřivová.

## License

MIT — see [LICENSE](./LICENSE)
