from cProfile import run
import logging
from logging import getLogger
import torch
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import numpy as np
import argparse
import torch.distributed as dist
import torch
from REC.data import LMDB_Image

from types import SimpleNamespace

from REC.data.dataset.data_utils import read_text, read_text_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings
from REC.data.dataset.data_utils.utils import *
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def build_text_item_content_for_dataload(config):
    """Build per-item text token features once (no text embeddings cached).
    Returns: (text_item_content, meta)
    - text_item_content: torch.LongTensor [item_num+1, text_feature_dim]
    """
    # config keys vary by project; use safe fallbacks
    root_data_dir = config['data_path']
    dataset = config['dataset']
    behaviors = config['behaviors']
    texts = config['texts']
    which_language = 'zh'
    max_seq_len = 21
    min_seq_len = 5
    num_words_title = 30

    args = SimpleNamespace(
        root_data_dir=root_data_dir,
        dataset=dataset,
        behaviors=behaviors,
        texts=texts,
        which_language=which_language,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        num_words_title=num_words_title,
        # for get_doc_input_bert
        num_words_abstract=0,
        num_words_body=0,
        news_attributes=['title'],
        NLP_model_load='chinese_bert_wwm',
    )

    # pick tokenizer name
    model_name = 'hfl/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    texts_path = os.path.join(args.root_data_dir, args.texts)
    # print(texts_path ,"\n", root_data_dir, "\n", dataset, "\n", behaviors, "\n", texts)
    before_item_id_to_dic, before_item_name_to_id = read_text_bert(
        texts_path, args, tokenizer, args.which_language
    )

    behaviors_path = os.path.join(args.root_data_dir, args.behaviors)
    item_num, item_id_to_dic, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = read_behaviors(
        behaviors_path,
        before_item_id_to_dic,
        before_item_name_to_id,
        args.max_seq_len,
        args.min_seq_len,
    )

    news_title, news_title_attmask, news_abstract, news_abstract_attmask, news_body, news_body_attmask = get_doc_input_bert(
        item_id_to_dic, args
    )
    item_content = np.concatenate(
        [x for x in [news_title, news_title_attmask,
                     news_abstract, news_abstract_attmask,
                     news_body, news_body_attmask] if x is not None],
        axis=1
    )
    text_item_content = torch.tensor(item_content, dtype=torch.long)

    meta = dict(
        item_num=item_num,
        users_train=users_train,
        users_valid=users_valid,
        users_test=users_test,
        users_history_for_valid=users_history_for_valid,
        users_history_for_test=users_history_for_test,
        args=args,
    )
    return text_item_content, meta

def run_loop(local_rank,config_file=None,saved=True):

    # configurations initialization
    config = Config(config_file_list=[config_file])

    ckpt_path = config['ckpt_path'] if config['ckpt_path'] else ' '
            
    device = torch.device("cuda", local_rank)
    config['device'] = device
    
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
                
    dataload = load_data(config)

    # Build text token samples once and attach to dataload so TrainDataset can emit txt_sample_items.
    # (This does NOT cache text embeddings; it only caches token ids + attention masks.)
    # try:
    text_item_content, text_meta = build_text_item_content_for_dataload(config)
    # print('text_item_content')
    # print(text_item_content)
    # print(text_meta)
    dataload.text_item_content = text_item_content
    dataload._text_meta = text_meta
    logger.info(set_color('[Text]: ', 'pink') + f'text_item_content shape={tuple(text_item_content.shape)}')
    # except Exception as e:
    #     logger.warning(set_color('[Text]: ', 'pink') + f'failed to build text_item_content: {e}')
    
    print("$$dataload:", dataload)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)               
    model = get_model(config['model'])(config, dataload)    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) 
    
        
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            #if 'item_embedding' not in k:      #only load trm for id
            #if 'visual_encoder' not in k:      #only load trm for modal
            #if 'visual_encoder' in k:           #only load encoder
            if True:                           #load all
            #if False:                           #load nothing
                state_dict[k] = v
        rt = model.load_state_dict(state_dict,strict=False)
        logger.info(rt)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        model = model

    
    #world_size = torch.distributed.get_world_size()
    world_size = 1
    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)                    
    logger.info(dataload)
    #logger.info(model.module) 
    logger.info(getattr(model, "module", model))



    trainer = Trainer(config,model)

    resume_file = "Saved/MOBERT4Rec/MOBERT4Rec-Jun-17-2026_00-00-39-epoch22.pth"
    trainer.resume_checkpoint(resume_file)
    
    init_seed(config['seed'], config['reproducibility'])
    
    print("train_loader:", len(train_loader))
    # ds = train_loader.dataset
    # x = ds[0]
    # print("dataset[0] type:", type(x))
    # print("dataset[0]:", x)

    
    print("valid_loader:", len(valid_loader))                
    best_valid_score, best_valid_result = trainer.fit(
        train_loader, valid_loader, saved=saved, show_progress=False
)

    #model evaluation
    test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=False)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=-1, type=str)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file
    
    torch.cuda.set_device(local_rank)
    #dist.init_process_group(backend='nccl')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")

    run_loop(local_rank = local_rank,config_file=config_file)
   
