import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
import json
import ast
import clip
import argparse
import clip_net.clip
# from transformers import CLIPTokenizer, CLIPTextModel
from clip.simple_tokenizer import SimpleTokenizer
import clip_net.clip
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip_net.clip.load("ViT-L/14@336px", device=device)
# ViT-B/32 1 x 77 x 512
# ViT-L/14@336px 1 x 77 x 768


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_q', type=str, default='dataset/val_q.json',
                        help='path to the json file containing your prediction')
    parser.add_argument('--train_q', type=str, default='dataset/train_q.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--test_q', type=str, default='dataset/test_q.json',
                        help='path to the json file containing the ground true')


    params = parser.parse_args()
    return params


def qst_feat_extract(qst):

    text = clip_net.clip.tokenize(qst).to(device)
    token_ids = text[0].cpu().tolist() 

    tokenizer = SimpleTokenizer()
    vocab = tokenizer.encoder
    decoder = {v: k for k, v in vocab.items()} 

    tokens = [decoder[token_id] for token_id in token_ids if token_id in decoder]

    start_token = "<|startoftext|>"
    end_token = "<|endoftext|>"
    padding_token = "!"

    valid_tokens = [token for token in tokens if token not in [start_token, end_token, padding_token]]

    mask = [1 if token not in [start_token, end_token, padding_token] else 0 for token in tokens]

    valid_length = len(valid_tokens)


    tokenized_text = clip.tokenize([qst]).to(device)  # shape: [1, 77]
    with torch.no_grad():
        text_features = model.encode_text(text)
        
        x = model.token_embedding(tokenized_text).type(model.dtype)  # [1, 77, 512]
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = model.transformer(x)  
        x = x.permute(1, 0, 2) 

        token_features = x     
    
    return token_features, tokens, valid_length


def QstCLIP_feat(samples, dst_qst_path):

    ques_vocab = ['<pad>']
    # ans_vocab = []
    max_length = 0
    i = 0
    for sample in samples:
        i += 1
        question = sample['question'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        question_id = sample['question_id']
        print("\n")
        print("question id: ", question_id)

        save_file = os.path.join(dst_qst_path, str(question_id) + '.npz')

        if os.path.exists(save_file):
            print(question_id, " is already exist!")
            continue

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        
        # print(len(question))
        question = ' '.join(question)
        # question = question[:]
        print(question)
        
        
        qst_feat, tokens, valid_length = qst_feat_extract(question)
        tokens = np.array(tokens)
        print(qst_feat.shape)

        if max_length < valid_length:
            max_length = valid_length
        print('current max length: ', max_length)

        qst_feat = qst_feat.to(torch.float16)
        qst_features = qst_feat.cpu().numpy()

        # np.save(save_file, qst_features)
        np.savez(save_file, qst_feat=qst_features, tokens=tokens)

    print("max_length: ", max_length)


if __name__ == "__main__":

    params = parse_opt()

    train_q = json.load(open(params.train_q, 'r'))
    val_q = json.load(open(params.val_q, 'r'))
    test_q = json.load(open(params.test_q, 'r'))

    question = train_q + val_q + test_q

    dst_qst_path = "./feats/word"

    QstCLIP_feat(question, dst_qst_path)

    
