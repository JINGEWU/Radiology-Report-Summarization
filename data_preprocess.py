# -*- coding: utf-8 -*-
# @Time : 2022/11/01 16:00
# @File : data_preprocess.py
# @Author: JW
"""
this scripts is used for preprocessing MIMIC-III data from NOTEEVENT.csv file, please make sure you have the access to MIMIC data

"""

import re
import argparse
import json
import pandas as pd 
import re
import tqdm
import codecs
from tqdm.auto import tqdm
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Text Summarization:testing')
parser.add_argument('--path', type=str, default='./NOTEEVENTS.csv')
parser.add_argument('--output_path', type=str, default='./MIMIC_full.json')
args = parser.parse_args()

def merge_multiple_pages(text):
    
    sub_start = '(Over)'
    sub_end = '(Cont)'

    index_list = []
    index1 = text.find(sub_start)
    index2 = text.find(sub_end)
    while (index1 != -1) and (index2 !=-1) and (index1<index2):
        index_list.append(index1)
        text = text[:index1-1]+'\n'+text[index2+7:] 
        index1 = text.find(sub_start)
        index2 = text.find(sub_end)

    return text

def section_text(text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section, 
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n ([A-Z ()/,-]+):\s', re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('medical condition')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    return sections, section_names, section_idx

def clean_text(text):
    """ Clean up the impression string.
    This mainly removes bullet numbers for consistency.
    """
    pos_2 = text.find('FINAL REPORT')
    pos_1 = text.find('MEDICAL CONDITION')
    
    text = text[pos_1:pos_2]+text[pos_2+12:] 
    text = text.strip().replace('\n', '')
    text = text.replace(r'(Cont)', '')
    text = text.replace(r'(Over)', '')
    text = text.replace(r'FINAL REPORT', '')
    # remove bullet numbers
    
    text = re.sub(r'^[0-9]\.\s+', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s[0-9]\.\s+', ' ', text)
    text = re.sub(r'^[0-9]\)\s+', '', text)
    text = re.sub(r'\s[0-9]\)\s+', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = re.sub(r'\[.*\]', '____', text)  
    text = re.sub(r'[0-9]+', '____', text)
def load_mimic_reports(data_radio):
    """ Load the MIMIC-CXR reports from the zip file, using a set of study and subject IDs.
    """
   
    id2data = dict()
    len_data = data_radio.shape[0]
    for i in tqdm.tqdm(range(len_data)):
        study_id = data_radio.iloc[i]['row_id']
        subject_id = data_radio.iloc[i]['subject_id']
        hadm_id = data_radio.iloc[i]['hadm_id']
        text = data_radio.iloc[i]['text']
        text = merge_multiple_pages(text)
        sections, section_names, section_idx = section_text(text) 

        for name in section_names:
            idx_name = text.lower().find(name)
            if idx_name>0:

                text = text[:idx_name]+text[idx_name+len(name)+2:]  

        text = clean_text(text)   
        id2data[str(study_id)] = {
            'study_id': str(study_id),
                'subject_id': str(subject_id),
                'hadm_id': str(hadm_id),
                'sentences': text 
        }
    return id2data

    return text

def find_heading(text):
    s = r'\s+[A-Z]+\:'  
    pattern = re.compile(s)  
    result = pattern.findall(text,text.find('MEDICAL CONDITION')) # find headings in the "FINAL REPORT"
    result = [i.replace('\n','') for i in result] # remove '\n'
    result = [i.replace(' ','') for i in result] # remove space
    return result

def load_json_data(file_path):
    data = None
    with codecs.open(file_path, encoding='utf-8') as rf:
        data = json.load(rf)
    return data

def save_json_array(lst, file_path, encoding='utf-8'):
    with codecs.open(file_path, 'w') as wf:
        json.dump(lst, wf)


if __name__ == "__main__":
    data = pd.read_csv(args.path,low_memory=False)
    radiology = data[data.category=='Radiology'][['row_id','subject_id','hadm_id','description','text']]     # extract all radiology reports
    data_process = load_mimic_reports(radiology)    
    save_json_array(data_process,args.output_path) 


