# OCR and Text Summarization

Simple python app for OCR and Text Summarization. It's a part of my learning about these subjects.
Tesseract was used for OCR, nltk and spaCy lib for Text Summarization.

# Requirements

You can found all necessary libs in `requirements.txt`.

- Create a conda env (put your env name where is "env")
```bash
conda create --name <env> --file requirements.txt
```

- Activate conda env
```bash
conda activate <env>
```

⚠️ Only **english** and **portuguese** languages are avaliable for this application. Since we're using spaCy lib for nlp processing, we need to download trained models, so you'll need do download it with the command `python -m spacy download <language id>` (check [this tutorial](https://spacy.io/usage/models)).

# Usage

```bash
python main.py --input samples/img_test1.webp --sentences 6

```

- `sentences`: Max value of sentences to include on summary

> The output is a html page with the original text taken from image using OCR and the best sentences highlighted

# Acknowledgements

This code was based on knowledge acquired in the course "[Text Summarization with Natural Language Processing](https://iaexpert.academy/courses/sumarizacao-de-textos-com-processamento-de-linguagem-natural/)".
