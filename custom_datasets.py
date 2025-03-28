import csv
from pickle import NONE
import torch
from transformers import BartTokenizer, BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from transformers.models.idefics.image_processing_idefics import valid_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ckpt = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_ckpt)
Bert_ckpt = 'bert-base-uncased'
Bert_tokenizer = BertTokenizer.from_pretrained(Bert_ckpt)
class GANBARTDataset(Dataset):
    def __init__(self, sentence1_list, sentence2_list, max_length=512):
        """
        Args:
            sentence1_list (list of str): List of sentence 1 strings.
            sentence2_list (list of str): List of sentence 2 strings.
            labels (list of int, optional): List of labels for the sentences (default is None).
            max_length (int, optional): Maximum token length for padding/truncation (default is 512).
        """
        self.sentence1_list = sentence1_list
        self.sentence2_list = sentence2_list
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence1_list)

    def __getitem__(self, idx):

        sentence = self.sentence1_list[idx]
        labels = self.sentence2_list[idx]

        # Tokenize the Lecture and Summary
        encoding = tokenizer(
            sentence, text_target = labels,  # using Lecture and Summary
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        bert_encoding = Bert_tokenizer(labels, # using Summary only
            padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        )
        #bert_fencoding = Bert_tokenizer(sentence, 
            #padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        #)
        # Get the tokenized inputs (input_ids, attention_mask)
        input_ids = encoding['input_ids'].squeeze(0)  # Shape (1, seq_len) -> remove batch dimension
        label = encoding['labels'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        #print(bert_encoding)
        bert_input_id = bert_encoding['input_ids'].squeeze(0)
        bert_mask = bert_encoding['attention_mask'].squeeze(0)

        #bert_finput_id = bert_fencoding['input_ids'].squeeze(0)
        #bert_fmask = bert_fencoding['attention_mask'].squeeze(0)


        return {'input_ids':input_ids , 
        'attention_mask': attention_mask , 
        'label': label, 
        'bert_input_id': bert_input_id, 
        'bert_mask': bert_mask 
        }

def data_augmentation(alist): # not yet implement
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer_en = AutoTokenizer.from_pretrained(model_name)
    print("Default Max Length:", tokenizer_en.model_max_length)
    tokenizer_fr = AutoTokenizer.from_pretrained(model_name, src_lang='ron_Latn')
    print("Default Max Length:", tokenizer_fr.model_max_length)
    model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model1.to(device)
    model2.to(device)
    target_list = ["fra_Latn","deu_Latn"]
    translated_list = []
    print(f"total:{len(alist)} articles")
    num = len(alist)
    for sentence in alist:
      print(f"Left with:{num} articles")
      with torch.no_grad():
        input_sentence = tokenizer_en(sentence,return_tensors="pt")
        input_sentence.to(device)
        translated_tokens = model1.generate(
          **input_sentence, forced_bos_token_id=tokenizer_en.convert_tokens_to_ids("ron_Latn"),max_length=1024
        )
        tr = tokenizer_en.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        tr_input = tokenizer_fr(tr,return_tensors="pt")
        tr_input.to(device)
        tr_translated_tokens = model2.generate(
          **tr_input, forced_bos_token_id=tokenizer_fr.convert_tokens_to_ids("en_Latn"),max_length=1024
        )
        back_tr = tokenizer_fr.batch_decode(tr_translated_tokens, skip_special_tokens=True)[0]
        translated_list.append(back_tr)
      num -= 1
    return translated_list



def create_dataset(is_argument = False,is_read=False,lecture_path=None, summary_path=None):
    # create datasets
    with open('./Datasets/generated_lectures.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
    data_set = list(zip(*rows))
    if is_argument: # not yet implement
        t_Lecture = list(data_set[3][:5])
        blist = data_augmentation(t_Lecture)
        with open("./Datasets/article.txt", "w") as file:
            for paragraph in blist:
                file.write(paragraph + "\n\n")
        with open("./Datasets/article.txt", "r") as file:
            read_article_list = file.read().split("\n\n")  # Split by empty lines (paragraphs)

    else: #get the list of data from Summary and Lecture
        t_Lecture = list(data_set[3][:800])
        t_summary = list(data_set[4][:800])
        v_Lecture = list(data_set[3][800:900])
        v_summary = list(data_set[4][800:900])
        test_Lecture = list(data_set[3][900:])
        test_summary = list(data_set[4][900:])
        # create and return datasets
        train_dataset = GANBARTDataset(t_Lecture, t_summary)
        validation_dataset = GANBARTDataset(v_Lecture, v_summary)
        test_dataset = GANBARTDataset(test_Lecture, test_summary)
        del data_set
        del t_Lecture
        del t_summary
        del v_Lecture
        del v_summary

        return train_dataset, validation_dataset, test_dataset

def test_case(): # not yet implement
    alist = ["hello how are you man?","I am not a small piglet."]
    blist = data_augmentation(alist)
    print(blist)
    with open("./Datasets/article.txt", "w") as file:
        for paragraph in blist:
            file.write(paragraph + "\n\n")

    with open("./Datasets/article.txt", "r") as file:
        read_article_list = file.read().split("\n\n")  # Split by empty lines (paragraphs)

    # Print the read list
    print("Read Article List:")
    for paragraph in read_article_list:
        print(paragraph)

