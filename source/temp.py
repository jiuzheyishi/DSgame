
from transformers import PegasusTokenizer,PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
from transformers import BartTokenizer,BartForConditionalGeneration
from settings import *
from utils import GetRouge,CountFiles
import os
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.module import Module

def ReadJson(i,dir,test=False):
    '''读取单个json文件（一个样本）'''
    import json

    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    if test:
        return js_data["text"]
    return js_data["text"],js_data["summary"]

def TestOneSeq(net,tokenizer,text, target=None):
    '''生成单个样本的摘要'''
    torch.cuda.empty_cache()
    net.eval()
    
    text = str(text).replace('\n', '')
    input_tokenized = tokenizer.encode(
        text,
        truncation=True, 
        return_tensors="pt",
        max_length=SOURCE_THRESHOLD
        ).to(DEVICE)
 
    summary_ids = net.generate(input_tokenized,
                                    num_beams=NUM_BEAMS,
                                    no_repeat_ngram_size=3,
                                    min_length=MIN_LEN,
                                    max_length=MAX_LEN,
                                    early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    score=-1
    if(target!=None):
        score=GetRouge(output[0],target)
    return output[0],score

def GetTextSum_BART():
    tokenizer=BartTokenizer.from_pretrained(PARAM_DIR+"bart", output_past=True)
    net=BartForConditionalGeneration.from_pretrained(PARAM_DIR+"bart", output_past=True)
    print("bart 加载完毕")
    return (net.to(DEVICE),tokenizer)

def GenSub(net,tokenizer,param_path=None):
    '''生成submission.csv'''
    import csv
    from tqdm import tqdm
    
    if(param_path!=None):
        net.load_state_dict(torch.load(param_path))
    res=[]
    for i in tqdm(range(1000)):
        text=ReadJson(i,DATA_DIR+"new_test",True)
        summary=TestOneSeq(net,tokenizer,text)[0]
        res.append([str(i),summary])
    
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   
        writer.writerows(res)
        
        
net,tkz=GetTextSum_BART()
GenSub(net,tkz)