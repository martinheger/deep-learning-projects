# Deep Learning Projects

**[English](README.md) | [Česky](README_cz.md)**

Three practical projects from a Deep Learning course, with code and notebooks: image classification using CNN (with a deployed live app), a custom transformer trained on Czech laws, and fine-tuning Gemma 3 for structured output.

At the end of this page there is also a write-up of an individual case study on Czech news classification (fine-tuned transformers vs. classical ML on 104k articles). It is a larger piece of work done outside the course. Its code and notebooks are not in this repository because the data is not public.

## 1. CNN image classifier and live app

[Try the live demo](https://huggingface.co/spaces/MartinH-01/Flowers)

Convolutional neural network (fastai) classifying images (flowers). The model is deployed as a publicly available Gradio app on Hugging Face Spaces: upload an image, and the app returns a prediction with class probabilities.

- `app.py` - source code of the Gradio app (same code running on HF Spaces)
- `notebooks/` - training and experiments (Fashion-MNIST baseline, flower classification, final app model)

The model (`model_cleansplit.pkl`, ~50 MB) is not included in the repo, since the app runs directly on HF Spaces where the model is hosted.

## 2. Custom transformer: tokenization and generation on Czech laws

Implementation of a GPT-style transformer (based on [Karpathy's "Let's build GPT" tutorial](https://github.com/karpathy/ng-video-lecture)) trained on a corpus of Czech laws, including experiments with tokenization and model size.

Key findings (see [shrnuti-vysledky.txt](./2-transformer-tokenization/shrnuti-vysledky.txt) for full analysis):

- Tested two levels of tokenization (simple vs. improved), vocabulary of approx. 200k subword tokens
- Achieved metrics: train loss approx. 2.4, val loss approx. 3.1, so the capacity of the simple transformer was practically exhausted on this dataset
- Examples of generated text for empty vs. specific prompts in [example-outputs/](./2-transformer-tokenization/example-outputs/). The model accurately mimics the style of legal texts, even though the content is hallucinated

```text
2-transformer-tokenization/
├── notebooks/            # tokenization (2 variants) + dataset preparation
├── example-outputs/      # samples of generated text
├── merged_zakony.md      # source corpus (merged legal texts)
└── shrnuti-vysledky.txt  # summary of experiments and conclusions
```

The notebook `SimpleTokenizationMergedZakony.ipynb` assumes a Google Colab environment (loading data from `/content/`). To run locally, the path must be adjusted.

## 3. Fine-tuning Gemma 3: structured text analysis

Fine-tuning the Gemma 3 model (via [unsloth](https://github.com/unslothai/unsloth)) for structured text analysis of song lyrics (alternative title, summary, mood evaluation) on a dataset of 200 lyrics by Czech artists (Karel Kryl, Karel Plíhal, Ivan Mládek, Svěrák & Uhlíř).

Key findings (see [shrnuti.txt](./3-gemma-finetune-mood-classifier/shrnuti.txt)):

- The model successfully learned to follow the required output structure
- On 100 training examples, the fine-tuned model did not significantly outperform the baseline model; the small dataset limited improvements
- The model tended to default to a safe middle value when evaluating mood
- Observed a shift towards English at higher generation temperatures

```text
3-gemma-finetune-mood-classifier/
├── gemma3_finetune_final.ipynb      # fine-tuning pipeline
├── Vyuziti_GPT_api_dataset.ipynb    # dataset preparation via GPT API
├── train_100.jsonl / test_99.jsonl  # training and test data
└── shrnuti.txt                      # summary of results and observations
```

---

## Case study (individual work): Czech news classification, classical ML vs. fine-tuned transformers

> **Note:** this section is a write-up only. The repository contains no code or notebooks for it, because the data is not public. I am happy to show and explain the code on request.

Individual work, done as a case study in a hiring process for a Czech technology company. Unlike the course projects above, this was a complete solo project: from data cleaning through systematic experiments to a written report.

**Task.** Classify short Czech sports news texts (headline + lead paragraph) into 24 categories.

**Results** (macro F1 on a stratified 20 % validation split):

| Approach | Macro F1 |
|---|---|
| **Transformer (FERNET-C5, fine-tuned)** | **0.91 to 0.96** (depending on the run) |
| Logistic regression (TF-IDF) | 0.865 |
| Logistic regression (bag of words) | 0.857 |
| Logistic regression (embeddings, untuned) | 0.854 |
| SVM (TF-IDF) | 0.843 |
| XGBoost (embeddings, untuned) | 0.806 |
| XGBoost (bag of words) | 0.775 |
| XGBoost (TF-IDF) | 0.758 |

**Data.** 111,218 articles in, 104,006 after cleaning (7,183 exact duplicates and 29 rows with identical text and conflicting labels removed). Heavily imbalanced: the largest category holds 41 % of the data, the smallest only 4 articles. The main metric is therefore macro F1.

**What I did**

1. **EDA and cleaning:** class distribution, text lengths, duplicates, conflicting labels, repeated uninformative texts.
2. **Classical models:** logistic regression, linear SVM and XGBoost on TF-IDF and bag-of-words features, tuned with Optuna, with adjustable class weighting and diagnostics by text length.
3. **Transformer configuration search:** five pretrained Czech models, learning rates and class-weighting strength (linear and logarithmic schemes) on a stratified sample, several seeds per configuration.
4. **Final training:** FERNET-C5 and RobeCzech on the full data with early stopping, confusion matrices, per-category F1 and error analysis.
5. **Embeddings experiment:** vectors from the pretrained (not fine-tuned) model as input to logistic regression and XGBoost.

**Key findings**

- **Class weighting decides whether transformer training is stable.** Early runs were very unstable (macro F1 between 0 and 0.6 across seeds). The cause was extreme weights on rare classes combined with a small batch: a single example of a rare class could dominate the gradient.
- **Models respond differently to the weighting scheme.** On the sample, FERNET-C5 scored 0.677 with log1p weights and 0.675 with linear weights, while RobeCzech went from 0.533 to 0.878.
- **The gap between the two best models is smaller than run-to-run noise.** Repeated runs gave FERNET-C5 0.914 to 0.955 and RobeCzech 0.929 to 0.947. GPU training with float16 is not bit-reproducible even with a fixed seed, so a single number is not enough.
- **XGBoost fails on sparse matrices.** With TF-IDF features it reached 0.76; with dense transformer embeddings it improved to 0.81 without any tuning.
- **A domain model is not necessarily the best one.** A model pretrained on news (FERNET-News) did worse than the general FERNET-C5.
- **The errors make sense.** The most frequent confusions are between related categories (athletics and Olympic Games, motorsport and Formula 1).

**Limitations.** The final training uses the same validation set for early stopping and for the reported number, so the transformer result is slightly optimistic. A three-way split ran into a category with four articles. Tuning ran on a sample because of GPU limits.

**Tools used:** Python, scikit-learn, XGBoost, Optuna, PyTorch, Hugging Face Transformers, Google Colab (GPU).

---

## Tech stack

Python, PyTorch, fastai, Gradio, Hugging Face Spaces, Hugging Face Transformers, unsloth, OpenAI API, scikit-learn, XGBoost, Optuna

## Authorship and context

**Projects 1 to 3:** group assignments for the course PřF:M7DataSP – Advanced Data Science Practicum (2025) at Masaryk University, based on the course repository [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) and the shared team repository [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team). In collaboration with Lucia Kajanová and Eva Kopřivová.

**Case study:** individual work, not connected to any course.

## License

MIT, see [LICENSE](./LICENSE)
