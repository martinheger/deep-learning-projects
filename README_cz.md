# Deep Learning Projects

**[English](README.md) | [Česky](README_cz.md)**

Tři praktické projekty z kurzu Deep Learning: obrazová klasifikace pomocí CNN (s nasazenou live appkou), vlastní transformer natrénovaný na českých zákonech a fine-tuning Gemma 3 na strukturovaný výstup.

## 1. CNN klasifikátor obrázků a živá appka

[Vyzkoušet živé demo](https://huggingface.co/spaces/MartinH-01/Flowers)

Konvoluční neuronová síť (fastai) klasifikující obrázky (květiny / Fashion-MNIST). Model je nasazený jako veřejně dostupná Gradio appka na Hugging Face Spaces: nahraješ obrázek, appka vrátí predikci s pravděpodobnostmi jednotlivých tříd.

- `app.py` - zdrojový kód Gradio appky (stejný kód, jaký běží na HF Spaces)
- `notebooks/` - trénování a experimenty (Fashion-MNIST baseline, klasifikace květin, finální model appky)

Model (`model_cleansplit.pkl`, ~50 MB) není součástí repa, appka běží nasazená přímo na HF Spaces, kde je model uložený.

## 2. Vlastní transformer: tokenizace a generování na českých zákonech

Implementace GPT-style transformeru (vycházející z [Karpathyho tutoriálu "Let's build GPT"](https://github.com/karpathy/ng-video-lecture)) natrénovaného na korpusu českých zákonů, s experimenty nad tokenizací a velikostí modelu.

Klíčová zjištění (viz [shrnuti-vysledky.txt](./2-transformer-tokenization/shrnuti-vysledky.txt) pro plný rozbor):

- Vyzkoušeny dvě úrovně tokenizace (jednoduchá vs. vylepšená), slovník přibližně 200k subword tokenů
- Dosažené hodnoty: train loss přibližně 2.4, val loss přibližně 3.1, kapacita jednoduchého transformeru byla na tomto datasetu prakticky vyčerpaná
- Ukázky generovaného textu na prázdný vs. konkrétní prompt v [example-outputs/](./2-transformer-tokenization/example-outputs/). Model věrně napodobuje styl zákonů, i když si obsah vymýšlí

```text
2-transformer-tokenization/
├── notebooks/            # tokenizace (2 varianty) + příprava datasetu ze zákonů
├── example-outputs/      # ukázky generovaného textu
├── merged_zakony.md      # zdrojový korpus (sloučené texty zákonů)
└── shrnuti-vysledky.txt  # shrnutí experimentů a závěrů
```

Notebook `SimpleTokenizationMergedZakony.ipynb` počítá s prostředím Google Colab (načítá data z `/content/`). Pro lokální spuštění je potřeba cestu upravit.

## 3. Fine-tuning Gemma 3: strukturovaná analýza textu

Fine-tuning modelu Gemma 3 (přes [unsloth](https://github.com/unslothai/unsloth)) na úkol strukturované analýzy písňových textů (alternativní název, shrnutí, hodnocení nálady) na datasetu 200 textů od českých interpretů (Karel Kryl, Karel Plíhal, Ivan Mládek, Svěrák a Uhlíř).

Klíčová zjištění (viz [shrnuti.txt](./3-gemma-finetune-mood-classifier/shrnuti.txt)):

- Model se úspěšně naučil dodržovat požadovanou strukturu odpovědi
- Na 100 trénovacích příkladech fine-tuned model nepřekonal výrazně baseline model, malý dataset limitoval zlepšení
- Model měl tendenci volit při hodnocení nálady bezpečný střed
- Pozorován posun k angličtině při vyšší teplotě generování

```text
3-gemma-finetune-mood-classifier/
├── gemma3_finetune_final.ipynb      # fine-tuning pipeline
├── Vyuziti_GPT_api_dataset.ipynb    # příprava datasetu přes GPT API
├── train_100.jsonl / test_99.jsonl  # trénovací a testovací data
└── shrnuti.txt                      # shrnutí výsledků a pozorování
```

## Tech stack

Python, PyTorch, fastai, Gradio, Hugging Face Spaces, unsloth, OpenAI API

## Autorství a kontext

Skupinová zadání v rámci kurzu PřF:M7DataSP – Praktikum z pokročilé datové vědy (2025) na Masarykově univerzitě, na základě repozitáře kurzu [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) a sdíleného týmového repozitáře [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team).

Spolupráce s Lucií Kajanovou a Evou Kopřivovou.

## Licence

MIT, viz [LICENSE](./LICENSE)
