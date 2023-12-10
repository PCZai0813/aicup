<p align="center">
  <h1>TEAM_4084</h1>
  <p>
    Rank: 27.00 (20)	Task1: 0.7444 (28)	Task2: 0.5961 (26)
    <br />
    <a href="https://github.com/PCZai0813/aicup/blob/main/main.ipynb"><strong>本專案的主要程式碼 »</strong></a>
    <br />
  </p>

</p>

### 分數
|  User   | Rank  | 子任務 1：病患隱私資訊擷取 | 子任務 2：時間資訊正規化 | 
|  :---:  | :---: | :---------------------:  | :--------------------: |
| VIPeter |27.00 (20)| 0.7444 (28)                | 0.5961 (26) |

### Detailed Results
| Coding Type | Precision | Recall | F-measure | Support |
|-------------|----------:|:------:|----------:|--------:|
| MEDICALRECORD | 0.7817796 | 0.9879518 | 0.8728563 | 747 |
| PATIENT | 0.7439024 | 0.8519553 | 0.7942708 | 716 |
| IDNUM | 0.9434844 | 0.9528302 | 0.9481342 | 2120 |
| DATE | 0.9087024 | 0.9511997 | 0.9294655 | 2459 |
| DOCTOR | 0.8808538 | 0.8310791 | 0.8552428 | 3327 |
| STREET | 0.9126984 | 0.6686047 | 0.771812 | 344 |
| CITY | 0.9530387 | 0.924933 | 0.9387755 | 373 |
| STATE | 0.987842 | 0.9789157 | 0.9833586 | 332 |
| ZIP | 0.8362069 | 0.8243626 | 0.8302425 | 353 |
| DEPARTMENT | 0.8746736 | 0.7995227 | 0.8354115 | 419 |
| HOSPITAL | 0.8664383 | 0.8447412 | 0.8554522 | 1198 |
| DURATION | 0.5 | 0.4166667 | 0.4545455 | 12 |
| TIME | 0.84375 | 0.5170213 | 0.641161 | 470 |
| AGE | 0.9761904 | 0.8039216 | 0.8817204 | 51 |
| ORGANIZATION | 0.4505495 | 0.5540541 | 0.4969697 | 74 |
| SET | 0.75 | 0.6 | 0.6666667 | 5 |
| LOCATION-OTHER | 1 | 0.1666667 | 0.2857143 | 6 |
| PHONE | 0 | 0 | 0 | 1 |
| Micro-avg. F| 0.8796526 | 0.8721458 | 0.8758831 | 13007 |
| Macro-avg. F| 0.7894505 | 0.7041348 | 0.744356 | 13007 |
|-------------|----------:|:------:|----------:|----------:|
|-------------|----------:|:------:|----------:|----------:|
| Temporal Type | Precision | Recall | F-measure | Support |
|-------------|----------:|:------:|----------:|----------:|
| DATE | 0.8199316 | 0.7795852 | 0.7992495 | 2459 |
| DURATION | 1 | 0.4166667 | 0.5882353 | 12 |
| TIME | 0.2757202 | 0.1425532 | 0.1879383 | 470 |
| SET | 1 | 0.6 | 0.75 | 5 |
| Micro-avg.| 0.7694091 | 0.6761711 | 0.7197832 | 2946 |
| Macro-avg.| 0.7739129 | 0.4847013 | 0.5960788 | 2946 |
|-------------|----------:|:------:|----------:|----------:|

 <br>
# 目錄

- [上手指南](#上手指南)
  - [專案實驗配置](#專案實驗配置)
  - [安裝步驟](#安裝步驟)
- [檔目錄說明](#檔目錄說明)
- [資料前處理](#資料前處理)
- [模型訓練](#模型訓練)
- [匯入驗證集並生成標註與後處理](#匯入驗證集並生成標註與後處理)

### 上手指南

“/PCZai0813/aicup”

###### 專案實驗配置

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
  <br>
  
5. 此處會將Checkpoint存入checkpoint資料夾
    ```py
    # 在每個 epoch 保存模型
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_list': train_loss_list,
    }, "/checkpoint/model_checkpoint_epoch_{}.pt".format(epoch + 1))
    print("Model saved at epoch {}".format(epoch + 1)
    ```

### 匯入驗證集並生成標註與後處理

1. 匯入驗證集
   ```py
   from datasets import load_dataset, Features, Value
   valid_data = load_dataset("csv", data_files="opendid_valid.tsv", delimiter='\t',
                              features = Features({
                                  'fid': Value('string'), 'idx': Value('int64'),
                                  'content': Value('string'), 'label': Value('string')}),
                                  column_names=['fid', 'idx', 'content', 'label'])
   valid_list= list(valid_data['train'])
   valid_list
   ```
   <br>
2. 生成標註與後處理
   生成結果會在answer資料夾
   ```py
   f = open("/answer/answer{}.txt".format(model_epoch), "w")
   BATCH_SIZE = 50
   for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
     with torch.no_grad():
       seeds = valid_list[i:i+BATCH_SIZE]
       outputs = sample_batch(model, tokenizer, input=seeds)
       for o in outputs:
         f.write(o)
         f.write('\n')
   f.close()
   ```

該項目簽署了MIT 授權許可，詳情請參閱 [LICENSE.txt](https://github.com/your_github_name/your_repository/blob/master/LICENSE.txt)



