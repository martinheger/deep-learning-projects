# Deep Learning Projects

**[English](README.md) | [Česky](README_cz.md)**

Tři praktické projekty z kurzu Deep Learning, s kódem a notebooky: klasifikace obrázků pomocí CNN (s nasazenou živou aplikací), vlastní transformer natrénovaný na českých zákonech a fine-tuning Gemma 3 pro strukturovaný výstup.

Na konci stránky je navíc popis samostatné case study o klasifikaci českých zpráv (fine-tuning transformerů proti klasickému ML na 104 tisících článků). Je to rozsáhlejší práce mimo kurz. Kód a notebooky k ní v repozitáři nejsou, protože data jsou neveřejná.

## 1. CNN klasifikátor obrázků a živá aplikace

[Vyzkoušet živé demo](https://huggingface.co/spaces/MartinH-01/Flowers)

Konvoluční neuronová síť (fastai) klasifikující obrázky (květiny). Model je nasazený jako veřejně dostupná Gradio aplikace na Hugging Face Spaces: nahrajete obrázek a aplikace vrátí predikci s pravděpodobnostmi tříd.

- `app.py` - zdrojový kód Gradio aplikace (stejný kód běží na HF Spaces)
- `notebooks/` - trénink a experimenty (baseline na Fashion-MNIST, klasifikace květin, finální model pro aplikaci)

Model (`model_cleansplit.pkl`, ~50 MB) není součástí repozitáře, protože aplikace běží přímo na HF Spaces, kde je model uložený.

## 2. Vlastní transformer: tokenizace a generování na českých zákonech

Implementace transformeru ve stylu GPT (podle [tutoriálu „Let's build GPT" Andreje Karpathyho](https://github.com/karpathy/ng-video-lecture)) natrénovaného na korpusu českých zákonů, včetně experimentů s tokenizací a velikostí modelu.

Hlavní zjištění (celý rozbor v [shrnuti-vysledky.txt](./2-transformer-tokenization/shrnuti-vysledky.txt)):

- Vyzkoušeny dvě úrovně tokenizace (jednoduchá a vylepšená), slovník přibližně 200 tisíc subword tokenů
- Dosažené metriky: train loss přibližně 2,4, val loss přibližně 3,1, kapacita jednoduchého transformeru tedy byla na tomto datasetu prakticky vyčerpaná
- Ukázky generovaného textu pro prázdný a konkrétní prompt jsou v [example-outputs/](./2-transformer-tokenization/example-outputs/). Model věrně napodobuje styl právních textů, i když obsah je vymyšlený

```text
2-transformer-tokenization/
├── notebooks/            # tokenizace (2 varianty) + příprava datasetu
├── example-outputs/      # ukázky generovaného textu
├── merged_zakony.md      # zdrojový korpus (sloučené texty zákonů)
└── shrnuti-vysledky.txt  # shrnutí experimentů a závěrů
```

Notebook `SimpleTokenizationMergedZakony.ipynb` předpokládá prostředí Google Colab (načítá data z `/content/`). Pro lokální spuštění je potřeba upravit cestu.

## 3. Fine-tuning Gemma 3: strukturovaná analýza textu

Fine-tuning modelu Gemma 3 (přes [unsloth](https://github.com/unslothai/unsloth)) pro strukturovanou analýzu textů písní (alternativní název, shrnutí, hodnocení nálady) na datasetu 200 textů českých autorů (Karel Kryl, Karel Plíhal, Ivan Mládek, Svěrák & Uhlíř).

Hlavní zjištění (viz [shrnuti.txt](./3-gemma-finetune-mood-classifier/shrnuti.txt)):

- Model se úspěšně naučil dodržovat požadovanou strukturu výstupu
- Na 100 trénovacích příkladech dotrénovaný model výrazně nepřekonal základní model; malý dataset omezil zlepšení
- Model měl při hodnocení nálady sklon volit bezpečnou střední hodnotu
- Při vyšších teplotách generování se objevoval posun k angličtině

```text
3-gemma-finetune-mood-classifier/
├── gemma3_finetune_final.ipynb      # pipeline fine-tuningu
├── Vyuziti_GPT_api_dataset.ipynb    # příprava datasetu přes GPT API
├── train_100.jsonl / test_99.jsonl  # trénovací a testovací data
└── shrnuti.txt                      # shrnutí výsledků a pozorování
```

---

## Case study (samostatná práce): klasifikace českých zpráv, klasické ML vs. fine-tuning transformerů

> **Poznámka:** tato sekce je pouze popis. Kód ani notebooky k ní v repozitáři nejsou, protože data jsou neveřejná. Kód rád ukážu a vysvětlím na vyžádání.

Samostatná práce, vznikla jako case study ve výběrovém řízení pro českou technologickou firmu. Na rozdíl od projektů z kurzu výše šlo o kompletní samostatný projekt: od čištění dat přes systematické experimenty po písemnou zprávu.

**Úloha.** Zařadit krátké české sportovní texty (titulek + perex) do jedné z 24 kategorií.

**Výsledky** (macro F1 na stratifikované validační části, 20 % dat):

| Přístup | Macro F1 |
|---|---|
| **Transformer (FERNET-C5, fine-tuning)** | **0,91 až 0,96** (podle běhu) |
| Logistická regrese (TF-IDF) | 0,865 |
| Logistická regrese (bag of words) | 0,857 |
| Logistická regrese (embeddingy, bez ladění) | 0,854 |
| SVM (TF-IDF) | 0,843 |
| XGBoost (embeddingy, bez ladění) | 0,806 |
| XGBoost (bag of words) | 0,775 |
| XGBoost (TF-IDF) | 0,758 |

**Data.** 111 218 článků na vstupu, po čištění 104 006 (odstraněno 7 183 úplných duplicit a 29 řádků se stejným textem a různou kategorií). Silně nevyvážené třídy: největší kategorie má 41 % dat, nejmenší jen 4 články. Hlavní metrikou je proto macro F1.

**Co jsem udělal**

1. **EDA a čištění:** rozložení tříd, délky textů, duplicity, konfliktní labely, opakující se neinformativní texty.
2. **Klasické modely:** logistická regrese, lineární SVM a XGBoost nad TF-IDF a bag-of-words reprezentací, laděné Optunou, s nastavitelnou silou vážení tříd a diagnostikou podle délky textu.
3. **Hledání konfigurace transformeru:** pět předtrénovaných českých modelů, learning rate a míra vážení tříd (lineární i logaritmické schéma) na stratifikovaném vzorku, více seedů na konfiguraci.
4. **Finální trénink:** FERNET-C5 a RobeCzech na celých datech s early stoppingem, matice záměn, F1 po kategoriích a rozbor chyb.
5. **Experiment s embeddingy:** vektory z předtrénovaného (nedotrénovaného) modelu jako vstup pro logistickou regresi a XGBoost.

**Hlavní zjištění**

- **Vážení tříd rozhoduje o stabilitě tréninku transformeru.** První běhy byly velmi nestabilní (macro F1 mezi seedy od 0 do 0,6). Příčinou byly extrémní váhy vzácných tříd v kombinaci s malým batchem: jediný příklad ze vzácné třídy dokázal ovládnout gradient.
- **Modely reagují na schéma vážení různě.** Na vzorku měl FERNET-C5 0,677 s vahami log1p a 0,675 s lineárními, zatímco RobeCzech se posunul z 0,533 na 0,878.
- **Rozdíl mezi dvěma nejlepšími modely je menší než šum mezi běhy.** Opakované běhy daly FERNET-C5 0,914 až 0,955 a RobeCzech 0,929 až 0,947. Trénink na GPU s float16 není bitově reprodukovatelný ani při pevném seedu, jedno číslo proto nestačí.
- **XGBoost na řídkých maticích selhává.** S TF-IDF dosáhl 0,76, s hustými embeddingy z transformeru se bez ladění zlepšil na 0,81.
- **Doménový model nemusí být nejlepší.** Model předtrénovaný na zpravodajství (FERNET-News) dopadl hůř než obecný FERNET-C5.
- **Chyby dávají smysl.** Nejčastější záměny jsou mezi příbuznými kategoriemi (atletika a olympijské hry, motosport a formule 1).

**Omezení.** Finální trénink používá stejnou validační množinu pro early stopping i pro reportované číslo, výsledek transformeru je proto mírně optimistický. Trojí dělení naráželo na kategorii se čtyřmi články. Ladění probíhalo na vzorku kvůli limitům GPU.

**Použité nástroje:** Python, scikit-learn, XGBoost, Optuna, PyTorch, Hugging Face Transformers, Google Colab (GPU).

---

## Použité technologie

Python, PyTorch, fastai, Gradio, Hugging Face Spaces, Hugging Face Transformers, unsloth, OpenAI API, scikit-learn, XGBoost, Optuna

## Autorství a kontext

**Projekty 1 až 3:** skupinové úkoly do předmětu PřF:M7DataSP – Advanced Data Science Practicum (2025) na Masarykově univerzitě, vycházejí z repozitáře kurzu [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) a ze sdíleného týmového repozitáře [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team). Ve spolupráci s Lucií Kajanovou a Evou Kopřivovou.

**Case study:** samostatná práce, nesouvisí s žádným kurzem.

## Licence

MIT, viz [LICENSE](./LICENSE)
