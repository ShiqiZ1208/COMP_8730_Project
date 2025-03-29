
from torch.utils.data import DataLoader
from torch.optim import AdamW
from Lora import BART_base_model, Lora_fine_tuning_BART 
from transformers import get_scheduler
from accelerate.test_utils.testing import get_backend
from tqdm.auto import tqdm
import torch
import os
import evaluate
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BertTokenizer,set_seed
from datasets import load_dataset
from custom_datasets import create_dataset

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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





def Btrain_model(n_epochs, minibatch_sizes):
    print(n_epochs, minibatch_sizes)
########################################## load the tokenizer and model ##################################################
    # load model ckpt from huggingface and use it to tokenizer
    BART_model_ckpt = 'facebook/bart-large-cnn'
    BA_tokenizer = BartTokenizer.from_pretrained(BART_model_ckpt)
    # if there is no model create a model using pretrain model from huggingface
    BaseG_model = BART_base_model(BART_model_ckpt)
    BART = Lora_fine_tuning_BART(BaseG_model, lora_r, lora_alpha, lora_dropout, lora_query,
                          lora_key, lora_value, lora_projection, lora_mlp, lora_head
                          )

########################################## create datasets ################################################################
    t_dataset, v_dataset, test_dataset = create_dataset() # load datasets
    train_dataloader = DataLoader(t_dataset, shuffle=False, batch_size=minibatch_sizes,worker_init_fn=lambda worker_id: np.random.seed(seed))
    print(len(train_dataloader))
    eval_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    print(len(eval_dataloader))
    rouge = evaluate.load("rouge") #load rouge socre evalutation
####################################### setting up training parameters ####################################################
    optimizerG = AdamW(BART.parameters(), lr=5e-5) # set up optimizer for Generator

    num_epochs = n_epochs # training epochs

    # set up learning schedualer for both discriminator and generator
    num_training_steps = num_epochs * len(train_dataloader)
    lr_schedulerG = get_scheduler(
        name="linear", optimizer=optimizerG, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device, _, _ = get_backend() # make sure the device is in gpu
    BART.to(device)


    print("\n=============================================start training==================================")

    print(f"\nNum_Epochs:{num_epochs}, Batch_size:{minibatch_sizes}")

########################################## training loop ################################################################
    progress_bar = tqdm(range(num_training_steps))


    epochs = 0
    BART.train()
    loss_record = []
    Rouge_record = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_rouge_1 = float('inf')
    epochs_without_improvement = 0
    patience = 3
    for epoch in range(num_epochs):
        batches = 0
        for batch in train_dataloader:

            # load information from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask =batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # calculate loss for both the CE loss from generated summary to true summary for BART
            # calculate the loss using fake summary and real label
            output = BART(input_ids=input_ids, attention_mask=attention_mask, labels = labels)

            # calculate final loss combine two loss before
            loss = output.loss 
            loss.backward()

            #update weight for generator(BART)
            optimizerG.step()
            lr_schedulerG.step()
            optimizerG.zero_grad()
            progress_bar.update(1)
            if batches % 10 == 0:
              loss_list = [loss]
              loss_record.append(loss_list)
            if batches % 20 == 0:
              print("\nEpoch:{: <5}| Batch:{: <5} train_loss:{: <5.4f}".format(epochs, batches, loss))
            batches +=1

        print(f"\n======================================Start Validation for Epoch: {epochs}==================================")
        BART.eval()
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
              outputs = BART(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
              genrated = BART.generate(input_ids=input_ids, attention_mask=attention_mask, labels = labels, max_length = 256)
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
        rouge_list = [a_rouge1, a_rouge2, a_rougeL, a_rougeLs]
        Rouge_record.append(rouge_list)
        print(f"\n======================================End Validation for Epoch: {epochs}==================================")


        epochs += 1
        torch.save(BART, f"./BARTmodel/lora_bart_{epoch}_{minibatch_sizes}")

    print("\n=============================================end training==================================")


