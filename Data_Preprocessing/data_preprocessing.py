import os

import numpy as np

def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}


def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案

    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)

    bounary , item_idx , temp_seq , seq_pairs = 0 , 0 , "" , []
    new_line_idx = 0
    for w_idx, word in enumerate(article):
        if word == '\n':
            new_line_idx = w_idx + 1
            if article[bounary:new_line_idx] == '\n':
                continue
            if temp_seq == "":
                temp_seq = "PHI:Null"
            sentence = article[bounary:new_line_idx].strip().replace('\t' , ' ')
            temp_seq = temp_seq.strip('\\n')
            seq_pair = f"{txt_name}\t{new_line_idx}\t{sentence}\t{temp_seq}\n"
            # seq_pair = special_tokens_dict['bos_token'] + article[bounary:new_line_idx] + special_tokens_dict['sep_token'] + temp_seq + special_tokens_dict['eos_token']
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\\n"
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    return seq_pairs

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder , tsv_output_path , num_processes=4):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())

    print("processing each medical file")

    all_seq_pairs = []
    for txt_name in txt_names:
        all_seq_pairs.extend(process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict))
    print(all_seq_pairs[:10])
    print("All medical file done")
    print("write out to tsv format...")
    with open(tsv_output_path , 'w' , encoding = 'utf-8') as fw:
        for seq_pair in all_seq_pairs:
            fw.write(seq_pair)
    print("tsv format dataset done")
    # return all_seq_pairs

anno_info_path = r"answer.txt"
report_folder = r"Validation_Release"
tsv_output_path = './train.tsv'
generate_annotated_medical_report_parallel(anno_info_path, report_folder, tsv_output_path, num_processes=4)
