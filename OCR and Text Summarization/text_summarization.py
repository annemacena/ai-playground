import re
import nltk
import string
import numpy as np
from nltk.cluster.util import cosine_distance
import networkx as nx
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import os

nltk.download('punkt')
nltk.download('stopwords')

class TextSummarization:
    """
    This class represents all steps involving text summarization

    ...

    Methods
    -------
    get_spacy_lang_id(text)
        Gets spaCy language id given text
        
    preprocessing(text, nlp_model)
        Preprocess given text, removing stopwords, punctuations and digits
        
    get_preprocessed_sentences(text)
        Separates sentences of given text and preprocessed text

    calculate_sentences_similarity(sentence1, sentence2)
        Calculates cosine similarity between sentences
        
    calculate_similarity_matrix(sentences)
        Calculates similarity matrix of a list of sentences
        
    summarize(text, qtd_sentences_summarized)
        Generates summary given text
        
    get_filename_incremented(text)
        Increments a number in the filename to not overwrite it

    save_summary(sentences, best_sentences)
        Saves text with the summary highlighted

    """

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english')) | set(nltk.corpus.stopwords.words('portuguese'))
        self.spacy_lang_ids = {
            "pt": "pt_core_news_sm",
            "en": "en_core_web_sm"
        }

    def get_spacy_lang_id(self, text):
        """
        Gets spaCy language id given text

        Args:
            text (String)

        Return:
            spaCy language id for trained model
        
        """

        # https://stackoverflow.com/a/66727355
        # https://spacy.io/usage/models
        def get_lang_detector(nlp, name):
                return LanguageDetector()

        nlp = spacy.load("en_core_web_sm")

        Language.factory("language_detector", func=get_lang_detector)        
        nlp.add_pipe('language_detector', last=True)

        language = nlp(text)._.language['language']

        return self.spacy_lang_ids[language]

    def preprocessing(self, text, nlp_model):
        """
        Preprocess given text, removing stopwords, punctuations and digits

        Args:
            text (String)
            nlp_model (Object): spaCy nlp object

        Return:
            Preprocessed text
        
        """
        text = text.lower()
        text = re.sub(r" +", ' ', text)

        document = nlp_model(text)
        tokens = []
        for token in document:
            tokens.append(token.lemma_)
        
        tokens = [word for word in tokens if word not in self.stopwords and word not in string.punctuation]
        formated_text = ' '.join([str(token) for token in tokens if not token.isdigit()])
        
        return formated_text

    def get_preprocessed_sentences(self, text):
        """
        Separates sentences of given text and preprocessed text

        Args:
            text (String)

        Return:
            List of sentences (string) of given texts and the list of sentences of tha same text preprocessed
        
        """

        nlp = spacy.load(self.get_spacy_lang_id(text))

        original_sentences = [sentenca for sentenca in nltk.sent_tokenize(text)]
        formated_sentences = [self.preprocessing(original_sentence, nlp) for original_sentence in original_sentences]

        return original_sentences, formated_sentences

    def calculate_sentences_similarity(self, sentence1, sentence2):
        """
        Calculates cosine similarity between sentences 

        Args:
            sentence1 (String)
            sentence2 (String)

        Return:
            Cosine similarity [-1, 1] between two sentences 
        
        """

        words1 = [word for word in nltk.word_tokenize(sentence1)]
        words2 = [word for word in nltk.word_tokenize(sentence2)]

        all_words = list(set(words1 + words2))

        word_vector1 = [0] * len(all_words)
        word_vector2 = [0] * len(all_words)

        for word in words1:
            word_vector1[all_words.index(word)] += 1
        for word in words2:
            word_vector2[all_words.index(word)] += 1
        
        return 1 - cosine_distance(word_vector1, word_vector2)

    def calculate_similarity_matrix(self, sentences):
        """
        Calculates similarity matrix of a list of sentences

        Args:
            sentences (List of String)

        Return:
            Similarity matrix of a list of sentences
        
        """

        qtd_sentences = len(sentences)
        similarity_matrix = np.zeros((qtd_sentences, qtd_sentences))

        for i in range(qtd_sentences):
            for j in range(qtd_sentences):
                if i == j:
                    continue
                similarity_matrix[i][j] = self.calculate_sentences_similarity(sentences[i], sentences[j])

        return similarity_matrix

    def summarize(self, text, qtd_sentences_summarized):
        """
        Generates summary given text

        Args:
            text (String)
            qtd_sentences_summarized (Integer): max value of sentences to include on summary

        Return:
            Original sentences, best sentences and ordered nodes 
        
        """

        original_sentences, formated_sentences = self.get_preprocessed_sentences(text)

        matriz_similaridade = self.calculate_similarity_matrix(formated_sentences)

        similarity_graph = nx.from_numpy_array(matriz_similaridade)
        nodes = nx.pagerank(similarity_graph)
        ordered_nodes = sorted(((nodes[i], node) for i, node in enumerate(original_sentences)), reverse=True)    

        best_sentences = []
        for i in range(qtd_sentences_summarized):
            best_sentences.append(ordered_nodes[i][1])

        return original_sentences, best_sentences, ordered_nodes

    def get_filename_incremented(self):
        """
        Increments a number in the filename to not overwrite it

        """
        
        if not os.path.exists("results"):
            os.makedirs("results")

        count = 0
        while os.path.exists(f"results/summary{count}.html"):
            count += 1
            
        return f"results/summary{count}.html"

    def save_summary(self, sentences, best_sentences = []):
        """
         Saves text with the summary highlighted

        Args:
            sentences (list): all sentences of a text
            best_sentences (list): best sentences of a text

        """

        text = ''

        for sentence in sentences:
            if sentence in best_sentences:
                text += str(sentence).replace(sentence, f"<mark>{sentence} </mark>")
            else:
                text += sentence + " "

        html = f'<html><body>{text}</body></html>'

        text_filename = self.get_filename_incremented()

        with open(text_filename, 'w', encoding='utf-8') as file:
            file.write(html)