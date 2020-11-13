import transformers 
import os

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "../models/model.bin"
TRAINING_FILE = "../input/imdb.csv"
TOKENIZER = transformers.BertTokenizer(os.path.join(BERT_PATH, "vocab.txt"), 
                                       do_lower_case=True
                                       )


