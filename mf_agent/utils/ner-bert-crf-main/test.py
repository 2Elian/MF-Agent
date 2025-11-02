import torch 
from torchcrf import CRF
from model import NerBertCrfModel
from modelscope import AutoTokenizer

device='cuda:3' if torch.cuda.is_available() else 'cpu'
labels = {'O': 0, 'PERSON-B': 1, 'PERSON-I': 2, 'ORGANIZATION-B': 3, 'ORGANIZATION-I': 4,
         'LOCATION-B': 5, 'LOCATION-I': 6, 'FIN_PRODUCT-B': 7, 'FIN_PRODUCT-I': 8,
         'CURRENCY-B': 9, 'CURRENCY-I': 10, 'DATE-B': 11, 'DATE-I': 12,
         'OPERATION-B': 13, 'OPERATION-I': 14, 'FIN_TERM-B': 15, 'FIN_TERM-I': 16, 'EVENT-B': 17, 'EVENT-I': 18}
labels_rev=dict((v,k) for k,v in labels.items())

tokenizer=AutoTokenizer.from_pretrained('/data1/nuist_llm/mf-agent/fin_ckpt/finbert2')
model=NerBertCrfModel(num_tags=len(labels)).to(device)
model.load_state_dict(torch.load('/data1/nuist_llm/mf-agent/ner-bert-crf-main/model.pt'))
model.eval()

def ner(s):
    tokens=[]
    for ch in s:
        ret=tokenizer([ch],add_special_tokens=False)
        tokens=tokens+ret.input_ids[0]
        
    bos_id=tokenizer.convert_tokens_to_ids(['[CLS]'])
    eos_id=tokenizer.convert_tokens_to_ids(['[SEP]'])

    input_ids=torch.tensor(bos_id+tokens+eos_id,dtype=torch.long).to(device)
    attn_mask=torch.tensor([1]*len(input_ids),dtype=torch.bool).to(device)
    type_ids=torch.tensor([0]*len(input_ids),dtype=torch.long).to(device)

    pred=model.predict(input_ids.unsqueeze(0),attn_mask.unsqueeze(0),type_ids.unsqueeze(0))
    print(pred)
    # ignore [CLS] and [SEP]
    input_ids=input_ids[1:-1]
    pred=pred[0][1:-1]
    start=None
    entity=''
    ner_result=[]
    for i in range(len(pred)):
        pred_label=labels_rev[pred[i]]
        pred_label_splits=pred_label.split('-')
        pred_label_first=pred_label_splits[0]
        pred_label_second='' if len(pred_label_splits)<=1 else pred_label_splits[1]
        if start is None and pred_label_second=='B': # entiry start
            start=i
            entity=pred_label_first
        elif start is not None and (pred_label_first!=entity or (pred_label_first==entity and pred_label_second=='B')):
            entity_value=[tokenizer.convert_ids_to_tokens([id])[0] for id in input_ids[start:i]]
            ner_result.append((start,i-1,entity,''.join(entity_value)))
            if pred_label_second=='B':
                start=i
                entity=pred_label_first
            else:
                start=None
                entity=''
    if start is not None:
        entity_value=[tokenizer.convert_ids_to_tokens([id])[0] for id in input_ids[start:]]
        ner_result.append((start,len(pred)-1,entity,''.join(entity_value)))
    return ner_result

s='在上海浦东发展银行公司网银中，如何通过“授权模式设置”和“操作员设置”实现针对特定账号和特定授权人的复杂授权流程？'
result=ner(s)
print(result)