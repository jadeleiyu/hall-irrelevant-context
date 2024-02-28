import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
import numpy as np
from ast import literal_eval
import hashlib
import json
import argparse


prompt_template_with_ctx = """Task instruction: answer the given question based on both your knowledge and the provided context. Ignore the context if it does not help you arrive at the answer. Keep the answer short and concise, ideally with less than 10 words.
Context: {}
Question: {}
The answer is:"""

prompt_template_no_ctx = """Task instruction: answer the given question based on your knowledge. Keep the answer short and concise, ideally with less than 10 words.
Question: {}
The answer is:"""


def main_no_context(args):
    # load LM and tokenizer
    device = torch.device('cuda')
    model_path = join(args.model_dir, args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    nq_dev_df = pd.read_csv(
        join(args.data_dir, 'nq-dev-with-ctx.csv'),
        converters={"answer": literal_eval}
    )

    # find questions that Llama can correctly answer without context 
    answers_no_ctx = []
    
    for _, row in tqdm(nq_dev_df.iterrows(), total=nq_dev_df.shape[0]):
        question = row['question']
        prompt_no_ctx = prompt_template_no_ctx.format(question)
        input_ids = tokenizer(prompt_no_ctx, return_tensors='pt').input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=50)
            answer_tokens = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        answers_no_ctx.append(answer_tokens)
        torch.cuda.empty_cache()
    
    nq_dev_df[f'{model_name} answer without context'] = answers_no_ctx
    
    is_correct_answer = []
    for i, row in nq_dev_df.iterrows():
        is_correct_i = 0
        model_answer = row[f'{model_name} answer without context']
        for true_answer in row['answer']:
            if true_answer in model_answer:
                is_correct_i = 1
        is_correct_answer.append(is_correct_i)        
    nq_dev_df[f'is {model_name} correct without context?'] = is_correct_answer

    nq_dev_df.to_csv(join(args.data_dir, 'nq-dev-with-ctx.csv'), index=False)


def main_with_context(args):
    # load LM and tokenizer
    device = torch.device('cuda')
    model_path = join(args.model_dir, args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    nq_dev_df = pd.read_csv(
        join(args.data_dir, 'nq-dev-with-ctx.csv'),
        converters={"answer": literal_eval, 'retrieved contexts': literal_eval}
    )
    nq_dev_df_factual = nq_dev_df.loc[
        nq_dev_df[f'is {model_name} correct without context?'] == 1
    ].reset_index(drop=True)

    answers = []
    is_correct_with_ctx = []
    for _, row in tqdm(nq_dev_df_factual.iterrows(), total=nq_dev_df_factual.shape[0]):
        question = row['question']
        row_answers = []
        row_is_correct_with_ctx = []
        for ctx in row['retrieved contexts']:
            prompt = prompt_template_with_ctx.format(ctx, question)
            
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids, max_new_tokens=50)
                answer_tokens = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            is_correct = 0
            for true_answer in row['answer']:
                if true_answer in model_answer:
                    is_correct = 1
            row_is_correct_with_ctx.append(is_correct)
            row_answers.append(answer_tokens)
        answers.append(row_answers)

    nq_dev_df_factual[f'{model_name} answers with context'] = answers
    nq_dev_df_factual[f'is {model_name} correct with context?'] = is_correct_with_ctx
    nq_dev_df_factual.to_csv(join(args.data_dir, 'nq-dev-with-ctx-answers.csv'), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/leiyu/projects/def-yangxu/leiyu/hall-irrelevant-context/data/', type=str)
    parser.add_argument('--model_dir', default='/home/leiyu/projects/def-yangxu/leiyu/LMs/', type=str)
    parser.add_argument('--model_name', default='Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--eval_with_ctx', default=True, type=bool)
    
    args = parser.parse_args()
    if args.eval_with_ctx:
        main_with_context(args)
    else:
        main_no_context(args)

    
    