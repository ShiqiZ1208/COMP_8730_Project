from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import csv



def data_augmentation(alist): # not yet implement
    device = "cuda"
    model_name = "facebook/nllb-200-distilled-600M"
    model_name2 = "facebook/nllb-200-distilled-600M"
    tokenizer_en = AutoTokenizer.from_pretrained(model_name)
    tokenizer_fr = AutoTokenizer.from_pretrained(model_name2)
    #print("Default Max Length:", tokenizer_en.model_max_length)
    model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model1.to(device)
    model2.to(device)
    translated = []
    lang_list = ["fra_Latn", "spa_Latn"]
    print(f"total:{len(alist)} articles")
    trans = []
    num = len(alist)
    for lecture in alist:
        print(f"Left with:{num} articles")
        trans.append(lecture)
        lines = lecture.splitlines()
        filtered_data = [lst for lst in lines if lst]
        for lang in lang_list:
            #print(lang)
            translate_list = []
            for sentence in filtered_data:
              with torch.no_grad():
                  input_sentence = tokenizer_en(sentence, max_length = 256, padding='max_length', return_tensors="pt")
                  input_sentence.to(device)
                  translated_tokens = model1.generate(
                    **input_sentence, forced_bos_token_id=tokenizer_en.convert_tokens_to_ids(lang),max_length=256
                  )
                  tr = tokenizer_en.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                  inputs_fr = tokenizer_fr(tr, return_tensors="pt", padding=True, truncation=True)
                  inputs_fr.to(device)
                  translated_tokens = model2.generate(
                  **inputs_fr,
                  forced_bos_token_id=tokenizer_fr.convert_tokens_to_ids("eng_Latn"),  # Target language is English (Latin script)
                  max_length=256
                  )
                  translated_text = tokenizer_fr.decode(translated_tokens[0], skip_special_tokens=True)
                  translate_list.append(translated_text)
            result = "\n".join(translate_list)
            trans.append(result)
        num -= 1
    return trans


def write_to_csv(lecture_list, summary_list):
    list_of_dicts = []
    for i in range(len(lecture_list)):
        list_of_dicts.append({
            "Lecture": lecture_list[i],
            "Summary": summary_list[i]
        })
    # Specify the CSV file path where you want to save the content
    file_path = "./Datasets/lecture_data_aug.csv"

    # Open the CSV file in write mode
    with open(file_path, mode='a', newline='') as file:
        # Define the fieldnames (column headers)
        fieldnames = ["ID", "Lecture", "Summary"]
        
        # Create a DictWriter object
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header (column names) to the CSV file
        file.seek(0, 2)  # Move the cursor to the end of the file
        if file.tell() == 0:
          writer.writeheader()

        # Use a loop to write each dictionary as a row in the CSV file
        for dictionary in list_of_dicts:
            writer.writerow(dictionary)  # Write each dictionary row by row

    print(f"have been written to {file_path}")


def data_aug(num):
  with open('./Datasets/generated_lectures.csv',newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
    rows = rows[1:]

  columns = list(zip(*rows))
  lecture = list(columns[3][0:num])
  summary = list(columns[4][0:num])
  del columns
  Lecture_list = data_augmentation(lecture)
  Summary_list = data_augmentation(summary)
  write_to_csv(Lecture_list, Summary_list)
