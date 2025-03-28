from flax.core.meta import B
from torch.utils.data import DataLoader
from torch.optim import AdamW
from Lora import BART_base_model, Lora_fine_tuning_BART, BERT_base_model, Lora_fine_tuning_BERT 
from transformers import get_scheduler
from accelerate.test_utils.testing import get_backend
from tqdm.auto import tqdm
import torch
import os
import evaluate
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BertTokenizer
from datasets import load_dataset
from custom_datasets import create_dataset

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load parameter for Lora fine tuning

# lora rank R
lora_r = 8
# lora_alpha
lora_alpha = 16
# lora dropout rate
lora_dropout = 0.05

# part of linear layer in base model that will be fine-tune with
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False





def train_model(n_epochs, minibatch_sizes, is_save, is_load=False, pathG=None, pathD=None):

########################################## load the tokenizer and model ##################################################
    # load model ckpt from huggingface and use it to tokenizer
    BART_model_ckpt = 'facebook/bart-large-cnn'
    BERT_model_ckpt = 'bert-base-uncased'
    BA_tokenizer = BartTokenizer.from_pretrained(BART_model_ckpt)
    BE_tokenizer = BertTokenizer.from_pretrained(BERT_model_ckpt)
    if is_load: # load the saved model from path
      NetG = torch.load(pathG, weights_only=False)
      NetD = torch.load(pathD, weights_only=False)
    else:
      # if there is no model create a model using pretrain model from huggingface
      BaseG_model = BART_base_model(BART_model_ckpt)
      NetG = Lora_fine_tuning_BART(BaseG_model, lora_r, lora_alpha, lora_dropout, lora_query,
                          lora_key, lora_value, lora_projection, lora_mlp, lora_head
                          )
      BaseD_model = BERT_base_model()
      NetD = Lora_fine_tuning_BERT(BaseD_model)

########################################## create datasets ################################################################
    t_dataset, v_dataset, test_dataset = create_dataset() # load datasets
    train_dataloader = DataLoader(t_dataset, shuffle=True, batch_size=minibatch_sizes)
    eval_dataloader = DataLoader(v_dataset, shuffle=True, batch_size=minibatch_sizes)
    rouge = evaluate.load("rouge") #load rouge socre evalutation
####################################### setting up training parameters ####################################################
    optimizerG = AdamW(NetG.parameters(), lr=5e-5) # set up optimizer for Generator
    optimizerD = AdamW(NetD.parameters(), lr=5e-5) # set up optimizer for Discrimnator

    num_epochs = n_epochs # training epochs

    # set up learning schedualer for both discriminator and generator
    num_training_steps = num_epochs * len(train_dataloader)
    lr_schedulerG = get_scheduler(
        name="linear", optimizer=optimizerG, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    lr_schedulerD = get_scheduler(
        name="linear", optimizer=optimizerD, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device, _, _ = get_backend() # make sure the device is in gpu
    NetG.to(device)
    NetD.to(device)



    print("\n=============================================start training==================================")

    print(f"\nNum_Epochs:{num_epochs}, Batch_size:{minibatch_sizes}")

########################################## training loop ################################################################
    progress_bar = tqdm(range(num_training_steps))


    epochs = 0
    NetD.train()
    NetG.train()
    for epoch in range(num_epochs):
        batches = 0
        for batch in train_dataloader:
            # Create real label and fake label for discrimnator
            oneslabel = torch.ones(minibatch_sizes)
            zeroslabel = torch.zeros(minibatch_sizes)
            tl = torch.vstack((oneslabel,zeroslabel))
            tl = torch.transpose(tl, 0, 1).to(device)
            fl = torch.vstack((zeroslabel,oneslabel))
            fl = torch.transpose(fl, 0, 1).to(device)

            # load information from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask =batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            bert_input_id = batch['bert_input_id'].to(device)
            bert_mask =batch['bert_mask'].to(device)

########################################## train the discriminator ######################################################

            # calculate loss from discriminator using summary from datasets and calcualate gradient
            output_td = NetD(bert_input_id, bert_mask, labels=tl)
            loss1 = output_td.loss
            loss1.backward()
            
            # generate fake summary using generator (BART)
            genrated = NetG.generate(input_ids=input_ids, attention_mask=attention_mask, labels = labels, max_length = 256).detach()
            G_data = []
            for i in range(minibatch_sizes):
                G_data.append(BA_tokenizer.decode(genrated[i],skip_special_tokens=True))
            #print(G_data)
            bert_encode = BE_tokenizer(G_data, 
            padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            bert_finput_id = bert_encode['input_ids'].squeeze(0)
            bert_fmask = bert_encode['attention_mask'].squeeze(0)

            bert_finput_id = bert_finput_id.to(device)
            bert_fmask = bert_fmask.to(device)

            # calculate loss using Bart generate summary with fake label
            output_fd = NetD(bert_input_id, bert_mask, labels=fl)
            loss2 = output_fd.loss
            loss2.backward()

            # calculate final loss and update the weight for discriminator(BERT)
            t_loss = (loss1+loss2)
            optimizerD.step()
            lr_schedulerD.step()
            optimizerD.zero_grad()

############################################ Training The Generator ####################################################
            NetG.zero_grad()
            # calculate loss for both the CE loss from generated summary to true summary for BART
            # calculate the loss using fake summary and real label
            output_g = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            output_fd = NetD(bert_input_id, bert_mask, labels=tl)

            # calculate final loss combine two loss before
            loss3 = output_g.loss + 0.25 * output_fd.loss
            loss3.backward()

            #update weight for generator(BART)
            optimizerG.step()
            lr_schedulerG.step()
            optimizerG.zero_grad()
            progress_bar.update(1)
            if batches % 40 == 0:
              print("\nEpoch:{: <5}| Batch:{: <5}| Gtrain_loss:{: <5.4f}| Dtrain_loss:{: <5.4f}".format(epochs, batches, loss3, t_loss))
            batches +=1

        print(f"\n======================================Start Validation for Epoch: {epochs}==================================")
        NetG.eval()
        for batch in eval_dataloader:
          t_loss = []
          t_rouge1 = []
          t_rouge2 = []
          t_rougeLs = []
          t_rougeL = []
          input_ids = batch['input_ids'].to(device)
          attention_mask =batch['attention_mask'].to(device)
          labels = batch['label'].to(device)
          with torch.no_grad():
              outputs = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
              genrated = NetG.generate(input_ids=input_ids, attention_mask=attention_mask, labels = labels, max_length = 256)
              G_data = []
              T_data = []
              for i in range(min(minibatch_sizes,len(genrated))):
                  G_data.append(BA_tokenizer.decode(genrated[i],skip_special_tokens=True))
                  T_data.append(BA_tokenizer.decode(labels[i],skip_special_tokens=True))
              r_score = rouge.compute(predictions=G_data, references=T_data)
              t_rouge1.append(r_score['rouge1'])
              t_rouge2.append(r_score['rouge2'])
              t_rougeL.append(r_score['rougeL'])
              t_rougeLs.append(r_score['rougeLsum'])
              t_loss.append(outputs.loss)
        a_rouge1 = sum(t_rouge1) / len(t_rouge1)
        a_rouge2 = sum(t_rouge2) / len(t_rouge2)
        a_rougeL = sum(t_rougeL) / len(t_rougeL)
        a_rougeLs = sum(t_rougeLs) / len(t_rougeLs)
        average_loss = sum(t_loss)/len(t_loss)
        print("\nEpoch:{: <5}| validation_loss:{: <5.4f}".format(epochs, average_loss))
        print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(a_rouge1, a_rouge2, a_rougeL, a_rougeLs))
        print(f"\n======================================End Validation for Epoch: {epochs}==================================")

        epochs += 1
        torch.save(NetG, f"./SaveModel/lora_bart_{epoch}_{minibatch_sizes}")

    print("\n=============================================end training==================================")
    
    if is_save:
        if is_load:
          torch.save(NetG, f"./SaveModel/continue_trained_lora_bart_{num_epochs}_{minibatch_sizes}")
          torch.save(NetD, f"./SaveModel/continue_trained_lora_bert_{num_epochs}_{minibatch_sizes}")
        else:
            torch.save(NetG, f"./SaveModel/lora_bart_{num_epochs}_{minibatch_sizes}")
            torch.save(NetD, f"./SaveModel/lora_bert_{num_epochs}_{minibatch_sizes}")


def model_predict(input_texts_file, pathG):
  model_ckpt = 'facebook/bart-large-cnn'
  tokenizer = BartTokenizer.from_pretrained(model_ckpt)
  model = torch.load(pathG, weights_only=False)

  model.eval()
  file_path = input_texts_file

  # Open the file in read mode and read the entire content
  with open(file_path, 'r') as file:
      content = file.read()

  print(f"Lecture:\n{content}")
  input_ids = tokenizer(content, truncation=True, padding='max_length', max_length=512, return_tensors= "pt")
  input_ids.to(device)
  output_ids = model.generate(**input_ids, max_length = 256)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  print(f"Summary:\n{output_text}")
  base_file_name = os.path.basename(input_texts_file)
  with open(f"./Summary/Summary_of_{base_file_name}", "w") as file:
    # Write the string to the file
    file.write(output_text)

  print(f"Summary of {base_file_name} created successfully!")



