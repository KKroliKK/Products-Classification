import pandas as pd
import natasha
from nltk.corpus import stopwords
import re
import numpy as np
from wikipedia2vec import Wikipedia2Vec
import fasttext
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import torch
from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights


SEED = 42


class NatashaTokenizer():
    stop_words = stopwords.words('russian')
    stop_words += ['.', ',', '"', '!', "''", '%', '«', '»', '“', '”', ':', '№', '=',
                '?', '(', ')', '-', '``', '@', '#', "'", '—', '/', '+', '&', '*',
                ':', ';', '_', '\\', '...', '\n', '$', '[', ']', '>', '<', '..']

    stop_tags = ['PUNCT', 'NUM']

    def __init__(self):
        self.segmenter = natasha.Segmenter()
        nat_emb = natasha.NewsEmbedding()
        self.morph_tagger = natasha.NewsMorphTagger(nat_emb)
        self.morph_vocab = natasha.MorphVocab()

    def tokenize(self, text, remove_numbers=True):
        try:
            doc = natasha.Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)

            def check(token):
                if token.pos in self.stop_tags:
                    return False
                if token.lemma in self.stop_words:
                    return False
                if len(token.lemma) < 3:
                    return False
                if re.search('\d', token.lemma) and remove_numbers:
                    return False
                return True

            tokens = [token.lemma for token in doc.tokens if check(token)]
        
        except:
            tokens = []

        return tokens


class Embedder():
    def __init__(self, use_fasttext=True, use_wiki2vec=False) -> None:
        assert (use_fasttext or use_wiki2vec) == True
        self.use_fasttext = use_fasttext
        self.use_wiki2vec = use_wiki2vec
        self.tokenizer = NatashaTokenizer()

        self.ft = fasttext.load_model('cc.ru.300.bin')
        if use_wiki2vec:
            self.wiki2vec = Wikipedia2Vec.load('ruwiki_20180420_300d.pkl')


    def vectorize(self, token):
        if self.use_fasttext:
            try:
                fast_text_vector = self.ft.get_word_vector(token)
            except KeyError:
                fast_text_vector = np.zeros((self.ft.get_dimension()))

        if self.use_wiki2vec:
            try:
                word2vec_vector = self.wiki2vec.get_word_vector(token)
            except KeyError:
                word2vec_vector = np.zeros((len(self.wiki2vec.get_word_vector('word'))))

        if self.use_fasttext and self.use_wiki2vec:
            return np.concatenate([word2vec_vector, fast_text_vector])
        elif self.use_fasttext:
            return fast_text_vector
        else:
            return word2vec_vector


    def get_sent_emb(self, sentence, remove_numbers=True):
        tokens = self.tokenizer.tokenize(sentence, remove_numbers)
        sent_emb = []
        for token in tokens:
            sent_emb.append(self.vectorize(token))

        if len(sent_emb) == 0:
            return np.zeros(300 * (self.use_fasttext + self.use_wiki2vec))
            
        return np.mean(sent_emb, axis=0)


def separate_train_test(
        df: pd.DataFrame,
        test_size=0.2,
        stratify_col=None   
    ) -> pd.DataFrame:

    df = df.copy()

    y_train, y_test = train_test_split(
                            df.index, 
                            test_size=test_size, 
                            stratify=df[stratify_col],
                            random_state=SEED
                        )

    df['train'] = df.index.isin(y_train)

    return df


def train_valid_test_split(
        df: pd.DataFrame,
        val_size=0.1,
        test_size=0.1,
        stratify_col: str=None
    ) -> pd.DataFrame:

    df = df.copy()

    train_idxs, val_idxs = train_test_split(
                            df.index, 
                            test_size=val_size, 
                            stratify=df[stratify_col],
                            random_state=SEED
                        )
    
    scaled_test_size = test_size / (1 - val_size)

    train_idxs, test_idxs = train_test_split(
                            train_idxs,
                            test_size=scaled_test_size, 
                            stratify=df[stratify_col].loc[train_idxs],
                            random_state=SEED
                        )

    df['train'] = df.index.isin(train_idxs)
    df['valid'] = df.index.isin(val_idxs)
    df['test'] = df.index.isin(test_idxs)

    return df


def get_embeddings(
        df: pd.DataFrame,
        column_name: str='product_name',
        file_name: str=None,
        folder: str='embeddings',
        use_fasttext=True,
        use_wiki2vec=False
    ):

    if not os.path.isdir(folder):
        os.makedirs(folder)

    embedder = Embedder(use_fasttext, use_wiki2vec)
    embeddings = []

    for title, index in zip(tqdm(df[column_name]), df.index):
        try:
            title_emb = embedder.get_sent_emb(title)
            embeddings.append(title_emb)
        except:
            print(title)
            print(index)
            break
    
    embeddings = np.array(embeddings)

    if file_name != None:
        path = os.path.join(folder, file_name)
        np.save(path, embeddings)
    
    return embeddings


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings.
    token_embeddings = model_output[0].detach().cpu()
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def make_features_transformers(
    df: pd.DataFrame, 
    col: str, 
    max_len: int,
    file_name: str=None,
    folder: str='embeddings',
    model_name: str='sberbank-ai/ruRoberta-large'
) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    text_features = []

    for sentence in tqdm(df[col]):
        encoded_input = tokenizer([sentence], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

        with torch.no_grad():
          model_output = model(input_ids=encoded_input['input_ids'].to(device))

        sentence_embeddings = list(mean_pooling(model_output, encoded_input['attention_mask']).numpy())
        text_features.extend(sentence_embeddings)

    text_features = np.array(text_features)

    if file_name != None:
        path = os.path.join(folder, file_name)
        np.save(path, text_features)

    return text_features


def make_features_cnn(
    df: pd.DataFrame,
    id_column: str,
    images_directory: str,
    filename_to_save: str=None,
    directory_to_save: str='./embeddings',
    model_architecture=resnet50,
    weights=ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2
) -> np.ndarray:
    model = model_architecture(weights, quantize=True)
    model.eval()
    preprocess = weights.transforms()

    feature_maps = {}
    for filename in tqdm(os.listdir(images_directory)):
        img = read_image(os.path.join(images_directory, filename))
        batch = preprocess(img).unsqueeze(0)
        out = model(batch).squeeze(0)
        feature_maps[filename[:-4]] = out

    pic_emb = [feature_maps[str(product_id)].numpy() 
               for product_id in df[id_column]]
    pic_emb = np.array(pic_emb)

    if filename_to_save != None:
        path = os.path.join(directory_to_save, filename_to_save)
        np.save(path, pic_emb)

    return pic_emb