import ast
import torch
import time
import json
import numpy as np

from pathlib import Path
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.models.clip import tokenize
from src.prompt_matcher import match_prompt
from src.models.vggish import wavfile_to_examples

import h5py
from tqdm import tqdm  

def load_clip_feat_vit_b32(clip_patch_path):
    memory_data = []
    vid_idx = {}

    with h5py.File(clip_patch_path, 'r') as hdf5_file:
        video_list = list(hdf5_file.keys()) 
        for idx, name in enumerate(tqdm(video_list, desc="Loading CLIP patch data")):
            if name not in vid_idx.keys():
                vid_idx[name] = idx
                memory_data.append(np.array(hdf5_file[name]))

    return memory_data, vid_idx

def load_frame_feat(clip_patch_path):
    memory_data = []
    vid_idx = {}

    with h5py.File(clip_patch_path, 'r') as hdf5_file:
        video_list = list(hdf5_file.keys())  
        for idx, name in enumerate(tqdm(video_list, desc="Loading frame-level feature")):
            if name not in vid_idx.keys():
                vid_idx[name] = idx
                memory_data.append(np.array(hdf5_file[name]))

    return memory_data, vid_idx

def load_clip_qst_feat(clip_qst_path):
    memory_data = []
    vid_idx = {}

    with h5py.File(clip_qst_path, 'r') as hdf5_file:
        video_list = list(hdf5_file.keys()) 
        for idx, name in enumerate(tqdm(video_list, desc="Loading CLIP Question data")):
            if name not in vid_idx.keys():
                vid_idx[name] = idx
                memory_data.append(np.array(hdf5_file[name]))

    return memory_data, vid_idx

def load_vggish_feat(vggish_path):
    memory_data = []
    vid_idx = {}

    with h5py.File(vggish_path, 'r') as hdf5_file:
        video_list = list(hdf5_file.keys()) 
        for idx, name in enumerate(tqdm(video_list, desc="Loading Audio data")):
            if name not in vid_idx.keys():
                vid_idx[name] = idx
                memory_data.append(np.array(hdf5_file[name]))

    return memory_data, vid_idx

def load_clip_word_feat(clip_word_path):
    memory_data = []
    word_mask = []
    vid_idx = {}

    start_token = "<|startoftext|>"
    end_token = "<|endoftext|>"
    padding_token = "!"

    with h5py.File(clip_word_path, 'r') as hdf5_file:
        video_list = list(hdf5_file.keys()) 
        for idx, name in enumerate(tqdm(video_list, desc="Loading CLIP word embedding")):
            if name not in vid_idx.keys():
                vid_idx[name] = idx
                qst_feat = hdf5_file[name]['qst_feat']
                tokens = np.array(hdf5_file[name]['tokens'])
                tokens = [token.decode('utf-8') if isinstance(token, bytes) else token for token in tokens]
                mask = [1 if token not in [start_token, end_token, padding_token] else 0 for token in tokens]
              
                word_mask.append(mask[:20])
                memory_data.append(np.array(qst_feat))

    return memory_data, word_mask, vid_idx

def load_data_into_memory(clip_patch_path, vggish_path, clip_qst_path, clip_word_path, frame_feat_path):

    vggish, vggish_idx = load_vggish_feat(vggish_path)
    frame_feat, frame_feat_idx = load_frame_feat(frame_feat_path)
    clip_qst, clip_qst_idx = load_clip_qst_feat(clip_qst_path)
    vit_patch, vit_patch_idx = load_clip_feat_vit_b32(clip_patch_path)
    clip_word, word_mask, clip_word_idx = load_clip_word_feat(clip_word_path)

    return vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx

qtype2idx = {
    'Audio': {'Counting': 0, 'Comparative': 1},
    'Visual': {'Counting': 2, 'Location': 3},
    'Audio-Visual': {'Existential': 4, 'Counting': 5, 'Location': 6,
                     'Comparative': 7, 'Temporal': 8}
}


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

class AVQA_dataset(Dataset):

    def __init__(self, 
                 config: dict,
                 mode: str,
                 audios_input, clip_input, clip_qst_input, clip_word_input, frame_feat_input,
                 transform: transforms.Compose = None,
    ):
        self.mode = mode 
        self.config = config
        self.type = config.type
        self.root = config.data.root

        self.clip_patch, self.clip_patch_idx = clip_input[0], clip_input[1]
        self.audio_feat, self.audio_feat_idx = audios_input[0], audios_input[1]
        self.clip_qst, self.clip_qst_idx = clip_qst_input[0], clip_qst_input[1]
        self.clip_word, self.word_mask, self.clip_word_idx = clip_word_input[0], clip_word_input[1], clip_word_input[2]
        self.frame_feat, self.frame_feat_idx = frame_feat_input[0], frame_feat_input[1]
        
        self.tokenizer = tokenize
        # self.bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        self.size = config.data.img_size
        self.sample_rate = config.data.frame_sample_rate
        
        annot = f'{self.mode}_annot'
        annot = eval(f"self.config.data.{annot}")
        annot = ROOT / self.root / annot
        
        with open(file=annot.as_posix(), mode='r') as f:
            self.samples = self.question_process(json.load(f))
        
        ans_quelen = self.get_max_question_length()
        self.answer_to_ix = ans_quelen['ans2ix']
        self.max_que_len = ans_quelen['max_que_len']
        self.config.num_labels = len(self.answer_to_ix)
        
        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)
        self.video_list = video_list
        self.video_len = 60 * len(video_list)
        self.cache = {}
        
        self.transform = transform if transform is not None \
            else transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                         std=IMAGENET_DEFAULT_STD),
                ])

    def __len__(self):
        return len(self.samples)
    
    def load_samples(self, sample):
        question_id = sample['question_id']

        # question preprocess
        labels = torch.tensor(data=[self.answer_to_ix[sample['anser']]], dtype=torch.long)
        ques_type = ast.literal_eval(sample['type'])
        qtype_label = torch.tensor([qtype2idx[ques_type[0]][ques_type[1]]], dtype=torch.long)
        if self.clip_qst is not None:
            idx = self.clip_qst_idx[str(question_id)]
            quest = self.clip_qst[idx].astype(np.float32)

        else:
            question = sample['question_content']
            quest = self.tokenizer(question, truncate=True).squeeze()
            prompt = self.tokenizer(sample['qprompt'], truncate=True).squeeze()

        name = sample['video_id']
        if self.frame_feat is not None:
            indx = self.frame_feat_idx[name]
            visual_CLIP_feat = self.frame_feat[indx].astype(np.float32)
            video = visual_CLIP_feat[::self.sample_rate] 

            indx = self.clip_patch_idx[name]
            visual_CLIP_feat = self.clip_patch[indx].astype(np.float32) 
            patch = visual_CLIP_feat[::self.sample_rate] 

        else:
            frame_dir = ROOT / self.root / self.config.data.frames_dir / name
            frame_path = sorted(list(frame_dir.glob('*.jpg')))[:60] 
            frame_path = frame_path[::self.sample_rate]
            video = torch.stack([
                self.transform(Image.open(frame_path[i]).convert('RGB'))
                for i in range(len(frame_path))
            ], dim=0)
            patch = None

        if self.audio_feat is not None:
            idx = self.audio_feat_idx[name]
            audio = self.audio_feat[idx].astype(np.float32)[::self.sample_rate]
        else:
            audio_dir = ROOT / self.root / self.config.data.audios_dir
            audio_path = audio_dir / f'{name}.wav'
            audio = wavfile_to_examples(audio_path.as_posix(), num_secs=60)
            audio = torch.from_numpy(audio)[::self.sample_rate]

        if self.clip_word is not None:
            idx = self.clip_word_idx[str(question_id)]
            word = self.clip_word[idx].astype(np.float32)
            word_mask = torch.tensor(self.word_mask[idx], dtype=torch.bool)
        
        data = {
            'quest': quest,
            'type': ques_type,
            'label': labels,
            'qtype_label': qtype_label,
            'video': video,
            'audio': audio,
            'word': word,
            'word_mask': word_mask,
            'name': name,
        }
        if patch is not None:
            data.update({'patch': patch})
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        batch = self.load_samples(sample)
        return batch

    def question_process(self, samples):
        for index, sample in enumerate(samples):
            question = sample['question_content'].lstrip().rstrip().split(' ')
            question[-1] = question[-1][:-1]  # delete '?'
            prompt = match_prompt(sample['question_content'], sample['templ_values'])
            
            templ_value_index = 0
            for word_index in range(len(question)):
                if '<' in question[word_index]:
                    question[word_index] = ast.literal_eval(sample['templ_values'])[templ_value_index]
                    templ_value_index = templ_value_index + 1
            samples[index]['question_content'] = ' '.join(question)  # word list -> question string
            samples[index]['qprompt'] = prompt
        return samples
    
    def get_max_question_length(self):
        ans_quelen = ROOT / self.root / self.config.data.ans_quelen
        if ans_quelen.exists():
            with open(file=ans_quelen.as_posix(), mode='r') as f:
                ans_quelen = json.load(f)
        else:
            ans_quelen = {}
            ans2ix = {}
            answer_index = 0
            max_que_len = 0

            # statistic answer in train split
            train_path = ROOT / self.root / self.config.data.train_annot
            valid_path = ROOT / self.root / self.config.data.valid_annot
            with open(file=train_path.as_posix(), mode='r') as f:
                samples = json.load(f)
            for sample in tqdm(samples):
                que_tokens = self.tokenizer(
                    sample['question_content'].lstrip().rstrip()[:-1]
                )
                que_len = len(torch.nonzero(que_tokens['input_ids']))
                if max_que_len < que_len:
                    max_que_len = que_len

                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1

            # statistic answer in val split
            with open(file=valid_path.as_posix(), mode='r') as f:
                samples = json.load(f)
            for sample in samples:
                que_tokens = self.tokenizer(
                    sample['question_content'].lstrip().rstrip()[:-1]
                )
                que_len = len(torch.nonzero(que_tokens['input_ids']))
                if max_que_len < que_len:
                    max_que_len = que_len

                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1

            # store it to a dict ,then to a json file
            save_path = ROOT / self.root / self.config.data.ans_quelen
            with open(file=save_path.as_posix(), mode='w') as f:
                ans_quelen['ans2ix'] = ans2ix
                ans_quelen['max_que_len'] = max_que_len
                json.dump(obj=ans_quelen, fp=f)
        return ans_quelen
