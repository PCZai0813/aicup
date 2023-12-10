

# AICUP


<!-- PROJECT LOGO -->
<br />

<p align="center">
  <h3 align="center">TEAM_4084</h3>
  <p align="center">
    Rank: 27.00 (20)	Task1: 0.7444 (28)	Task2: 0.5961 (26)
    <br />
    <a href="https://github.com/PCZai0813/aicup/blob/main/main.ipynb"><strong>本專案的主要程式碼 »</strong></a>
    <br />
  </p>

</p>
 
## 目錄

- [上手指南](#上手指南)
  - [開發前的配置要求](#開發前的配置要求)
  - [安裝步驟](#安裝步驟)
- [檔目錄說明](#檔目錄說明)
- [資料前處理](#資料前處理)
- [模型訓練](#模型訓練)
- [使用到的框架](#使用到的框架)
- [版本控制](#版本控制)

### 上手指南

“/your_github_name/your_repository”



###### 開發前的配置要求

1. 硬體環境<br>
   - 處理器（CPU）：Intel i5 13600K<br>
   - 顯示卡（GPU）：NVIDIA GeForce 3060ti
2. 程式語言<br>
   - Python 3.11.5
3. 套件<br>
   - Conda、PyTorch、Transformers、Datasets、Random、PEFT、TQDM、Matplotlib
###### **安裝步驟**

1. Clone the repo

```sh
git clone https://github.com/PCZai0813/aicup.git
```

### 檔目錄說明
eg:

```
filetree 
├── ARCHITECTURE.md
├── LICENSE.txt
├── README.md
├── /account/
├── /bbs/
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── manage.py
├── /oa/
├── /static/
├── /templates/
├── useless.md
└── /util/

```





### 資料前處理

1. 請將資料集放入 /Data_Preprocessing/dataset
2. 請將資料集標註檔放入 /Data_Preprocessing/answer
3. 運行此程式碼 /Data_Preprocessing/data_preprocessing.py
4. 最後會輸出 output.tsv 在 /Data_Preprocessing

### 模型訓練

1. 請使用 main.ipynb
   <br>
2. 此處可以修改預訓練模型，這段程式碼使用的是EleutherAI/pythia-160m-deduped
   ```py
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    plm = "EleutherAI/pythia-160m-deduped"
    
    bos = '<|endoftext|>'
    eos = '<|END|>'
    pad = '<|pad|>'
    sep ='\n\n####\n\n'
    
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}
    
    tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.padding_side = 'left'
    ```
   <br>
3. 此程式碼會將經過前處理的資料集匯入成dataset，此處使用了四個資料集
   ```py
   from datasets import load_dataset, Features, Value, concatenate_datasets
   dataset1 = load_dataset("csv", data_files="opendid_set1.tsv", delimiter='\t',
                           features = Features({
                                  'fid': Value('string'), 'idx': Value('int64'),
                                  'content': Value('string'), 'label': Value('string')}),
                           column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
    
   dataset2 = load_dataset("csv", data_files="opendid_set2.tsv", delimiter='\t',
                           features = Features({
                                  'fid': Value('string'), 'idx': Value('int64'),
                                  'content': Value('string'), 'label': Value('string')}),
                           column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
   dataset_fake = load_dataset("csv", data_files="fake.tsv", delimiter='\t',
                           features = Features({
                                  'fid': Value('string'), 'idx': Value('int64'),
                                  'content': Value('string'), 'label': Value('string')}),
                           column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
   dataset_Validation = load_dataset("csv", data_files="Validation_opendid.tsv", delimiter='\t',
                           features = Features({
                                  'fid': Value('string'), 'idx': Value('int64'),
                                  'content': Value('string'), 'label': Value('string')}),
                           column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
    
   # 組合四個資料集
   dataset = concatenate_datasets([dataset1['train'], dataset2['train'], dataset_fake['train'], dataset_Validation['train']])
   ```
   <br>
4. 此處可自行決定是否需要LoRA，可自行移除註解符號
   ```py
   #from peft import get_peft_model, LoraConfig, TaskType
   #peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
   #model = get_peft_model(model, peft_config)
   #model.print_trainable_parameters()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```
### 使用到的框架

- [xxxxxxx](https://getbootstrap.com)
- [xxxxxxx](https://jquery.com)
- [xxxxxxx](https://laravel.com)


### 版本控制

該專案使用Git進行版本管理。您可以在repository參看當前可用版本。

### 作者

xxx@xxxx

 *您也可以在貢獻者名單中參看所有參與該專案的開發者。*

### 版權說明

該項目簽署了MIT 授權許可，詳情請參閱 [LICENSE.txt](https://github.com/your_github_name/your_repository/blob/master/LICENSE.txt)



