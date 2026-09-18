import torch
from torch import nn
from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
from REC.model.load_lora import load_model #modified for lora
# from REC.model.load import load_model
import pickle
from REC.model.encoders_lora import Text_Encoder,  Bert_Encoder, UserEncoder_bert4rec #modified for lora
# from REC.model.encoders import Text_Encoder,  Bert_Encoder, UserEncoder_bert4rec
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, BertConfig
from types import SimpleNamespace

from REC.data.dataset.data_utils import read_text, read_text_bert, get_doc_input_bert, \
    read_behaviors, BuildTrainDataset, eval_model, get_item_embeddings
from REC.data.dataset.data_utils.utils import *
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F

from REC.model.rqvae import ResidualVectorQuantizer
from REC.model.vqvae import VectorQuantizer

class MOBERT4Rec(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(MOBERT4Rec, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  # same as embedding_size
        self.inner_size = self.hidden_size*config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.mask_ratio = config['mask_ratio']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.initializer_range = config['initializer_range']
        self.device = config['device']
        self.modal_fusion = config['modal_fusion']
        # load dataset info
        self.item_num = dataload.item_num
        
        self.mask_token = self.item_num
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self._compute_cursor = 0

        # define layers and loss
        self.visual_encoder = load_model(config=config)
        
        # modified for text encoder
        #bert_model_load = 'https://huggingface.co/hfl/chinese-bert-wwm-ext'
        bert_model_load = 'hfl/chinese-bert-wwm-ext'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_load)
        bert_config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
        nlp_model = BertModel.from_pretrained(bert_model_load, config=bert_config)
        # self.use_text_online = config.get('use_text_online', False) # set text encoder
        # if self.use_text_online:
        #     self.image_weight = float(config.get('image_weight', 0.5))
        #     self.text_weight = float(config.get('text_weight', 0.5))
        #     self.text_max_len = int(config.get('text_max_len', 16))
        #     self.tokenizer = AutoTokenizer.from_pretrained(config.get('text_model_name', 'bert-base-uncased')) # 从这里开始应该对着BERT4Rec_txt的run.py看一下
        #     nlp = AutoModel.from_pretrained(
        #         config.get('text_model_name', 'bert-base-uncased')
        #     ).to(self.device)
        #     self.txt_encoder = Text_Encoder(
        #     nlp_model=nlp,
        #     item_embedding_dim=self.hidden_size,      # 256
        #     word_embedding_dim=nlp.config.hidden_size
        # ).to(self.device)
        args = SimpleNamespace(
        embedding_dim=int(config['embedding_size']),     # YAML: embedding_size
        num_words_title=int(config['text_max_len']),      # YAML: text_max_len（先映射成 title 长度）
        num_words_abstract=0,
        num_words_body=0,
        news_attributes=['title'],                                # 最小先只用 title
        word_embedding_dim=nlp_model.config.hidden_size
)
        self.args = args
        self.max_seq_length_txt = 2
        self.nlp_encoder = Bert_Encoder(args=args, nlp_model=nlp_model)
        #self.txt_encoder.eval()
        #for p in self.txt_encoder.parameters():
        #    p.requires_grad = False
        # modified for text encoder ends here

        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # parameters initialization
        self.position_embedding.weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.trm_encoder.apply(self._init_weights)
        self.LayerNorm.bias.data.zero_()
        self.LayerNorm.weight.data.fill_(1.0)

        # modified for VQ-VAE / RQ-VAE ablation
        # quantizer_type choices: "none", "vqvae", "rqvae"
        self.quantizer_type = config['quantizer_type']

        self.use_quantizer = self.quantizer_type in ['vqvae', 'rqvae']

        if self.use_quantizer:
            self.quantizer_loss_weight = float(config['quantizer_loss_weight'])

        if self.quantizer_type == 'rqvae':
            self.rqvae_num_codebooks = int(config['rqvae_num_codebooks'])
            self.rqvae_codebook_size = int(config['rqvae_codebook_size'])

            self.quantizer = ResidualVectorQuantizer(
                dim=self.hidden_size,
                num_codebooks=self.rqvae_num_codebooks,
                codebook_size=self.rqvae_codebook_size
            )

            self.quantizer_fusion_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

        elif self.quantizer_type == 'vqvae':
            self.vqvae_codebook_size = int(config['vqvae_codebook_size'])
            self.vqvae_commitment_weight = float(config['vqvae_commitment_weight'])

            self.quantizer = VectorQuantizer(
                dim=self.hidden_size,
                codebook_size=self.vqvae_codebook_size,
                commitment_weight=self.vqvae_commitment_weight
            )

            self.quantizer_fusion_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # __init__ 里，冻结逻辑之前加
        self.text_item_content = torch.tensor(
            dataload.text_item_content, dtype=torch.long
        ).to(self.device)  # [item_num+1, token_len]

        # 预计算所有 item 的 BERT CLS 特征，固定不更新
        self.nlp_encoder = self.nlp_encoder.to(self.device)
        self.nlp_encoder.eval()
        with torch.no_grad():
            cls_features = []
            all_tokens = self.text_item_content
            for i in range(0, len(all_tokens), 256):
                batch = all_tokens[i:i+256]
                num_words = self.args.num_words_title
                input_ids = batch[:, :num_words]
                attn_mask = batch[:, num_words:num_words*2]
                out = self.nlp_encoder.text_encoders['title'].nlp_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask
                )[0][:, 0]  # [B, 768] CLS token
                cls_features.append(out.cpu())
            self.bert_item_features = torch.cat(cls_features, dim=0).to(self.device)
            self.bert_item_features[0].zero_()
            # shape: [item_num+1, 768]，固定不更新

        print("===== TEXT DEBUG =====")

        print("text_item_content shape:")
        print(self.text_item_content.shape)

        print("item 1 tokens:")
        print(self.text_item_content[1][:30])

        print("item 10 tokens:")
        print(self.text_item_content[10][:30])

        print("item 100 tokens:")
        print(self.text_item_content[100][:30])

        print("bert_item_features shape:")
        print(self.bert_item_features.shape)

        print("feature norms:")
        print(self.bert_item_features.norm(dim=1)[:20])

        # ===== 关键：把全量训练改成非全量训练 =====
        for p in self.parameters():
            p.requires_grad = False

        # 只训练推荐系统部分
        for p in self.position_embedding.parameters():
            p.requires_grad = True

        for p in self.trm_encoder.parameters():
            p.requires_grad = True

        for p in self.LayerNorm.parameters():
            p.requires_grad = True
        
        # __init__ 冻结逻辑之后加
        for p in self.nlp_encoder.text_encoders['title'].fc.parameters():
            p.requires_grad = True

        # partial fine-tune BERT: unfreeze last 2 layers
        bert = self.nlp_encoder.text_encoders['title'].nlp_model

        for layer in bert.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        # modified for VQ-VAE / RQ-VAE
        if self.use_quantizer:
            for p in self.quantizer.parameters():
                p.requires_grad = True

            for p in self.quantizer_fusion_proj.parameters():
                p.requires_grad = True

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



    def reconstruct_test_data(self, item_seq):

        padding = torch.full(size=(item_seq.size(0),1), fill_value=self.mask_token, dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding), dim=-1)  # [B max_len+1]

        return item_seq
    


    def freeze_lora_only(self):
        """
        Freeze only LoRA parameters.
        Keep Transformer, RQ-VAE, projection, fc, etc. unchanged.
        """
        frozen = 0
        total_lora = 0

        for name, p in self.named_parameters():
            if "lora" in name.lower():
                total_lora += p.numel()
                if p.requires_grad:
                    frozen += p.numel()
                p.requires_grad = False

        print(f"[INFO] Frozen LoRA parameters: {frozen}/{total_lora}")


    def print_trainable_parameters(self):
        trainable = 0
        total = 0

        print("===== Trainable Parameters After Freezing LoRA =====")
        for name, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                print(name, tuple(p.shape))

        print(f"Trainable params: {trainable}/{total} = {100 * trainable / total:.4f}%")



    def apply_quantizer_feature(self, item_emb):
        if not self.use_quantizer:
            return item_emb, item_emb.new_tensor(0.0)

        quantized_emb, quantizer_indices, quantizer_loss = self.quantizer(item_emb)

        enhanced_emb = torch.cat([item_emb, quantized_emb], dim=-1)
        delta = self.quantizer_fusion_proj(enhanced_emb)

        enhanced_emb = item_emb + 0.1 * delta

        return enhanced_emb, quantizer_loss


   
    def forward(self, input):
        #masked_sequence, instance, neg_items, masked_index
        input_ids, items_modal, txt_sample_items, masked_index = input
     
        batch_size = masked_index.shape[0]
        if self.modal_fusion in ('img', 'fused'): 
            item_emb_img = self.visual_encoder(items_modal.flatten(0,1)).view(batch_size, -1, 3, self.hidden_size) # image embedding

        # print("txt_sample_items.shape =", txt_sample_items.shape)

        # if self.modal_fusion in ('txt', 'fused'):
        #     B, S3, C = txt_sample_items.shape
        #     txt_flat = txt_sample_items.view(B * S3, C)
        #     with torch.autocast('cuda', enabled=False):  # 强制 fp32
        #         item_emb_txt_all = self.nlp_encoder(txt_flat)
        
        # # print("item_emb_txt_all.shape =", item_emb_txt_all.shape)
        #     item_emb_txt = item_emb_txt_all.view(B, S3//3, 3, self.args.embedding_dim)

        if self.modal_fusion in ('txt', 'fused'):
            B, S3 = txt_sample_items.shape

            if not hasattr(self, "_printed_txt_batch"):
                self._printed_txt_batch = True
                print("===== TXT SAMPLE DEBUG =====")
                print("txt_sample_items shape:", txt_sample_items.shape)
                print("txt_sample_items min/max:", txt_sample_items.min().item(), txt_sample_items.max().item())
                print("txt_sample_items first row FULL:", txt_sample_items[0])
                print("nonzero count first row:", (txt_sample_items[0] != 0).sum().item())
                print("item_num:", self.item_num)

            tokens = self.text_item_content[txt_sample_items]  # [B, S3, token_len]
            txt_flat = tokens.view(B * S3, -1)

            with torch.autocast(device_type='cuda', enabled=False):
                item_emb_txt_all = self.nlp_encoder(txt_flat)
            item_emb_txt_all = F.normalize(item_emb_txt_all, dim=-1)
            item_emb_txt = item_emb_txt_all.view(B, S3//3, 3, self.hidden_size)
            item_emb_txt = item_emb_txt * (txt_sample_items.view(B, S3//3, 3).unsqueeze(-1) != 0).float()

        # print("item_emb_img:", item_emb_img.shape)
        # print("item_emb_txt:", item_emb_txt.shape)

        if self.modal_fusion == 'img':
            item_emb = item_emb_img
        elif self.modal_fusion == 'txt':
            item_emb = item_emb_txt
        else:  # fused
            item_emb = 0.5 * item_emb_img + 0.5 * item_emb_txt

        # RQ-VAE feature augmentation
        item_emb, quantizer_loss = self.apply_quantizer_feature(item_emb)
    

        # print("item_emb shape right before slicing:", item_emb.shape)
        
        input_items_embs = item_emb[:,:, 0]  #[batch, max_seq_len+1, dim]
        pos_items_embs = item_emb[:,:, 1]   #[batch, max_seq_len+1, dim]
        neg_items_embs = item_emb[:,:, 2]

        position_ids = torch.arange(end=input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embedding = self.position_embedding(position_ids)

        # print("input_items_embs:", input_items_embs.shape)
        # print("position_embedding:", position_embedding.shape)

        input_emb = input_items_embs + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(input_ids)
        
        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
        output_embs = output_embs[-1]
        
        indices = torch.where(masked_index != 0)
        batch = masked_index.shape[0]
        seq_output = output_embs[indices]
        pos_items_emb = pos_items_embs[indices]
        neg_items_emb = neg_items_embs[indices]


        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B*mask_len]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B*mask_len]

        # loss = - (torch.log(1e-6 + torch.sigmoid(pos_score - neg_score))).sum(-1) 
   
        # return loss/batch

        rec_loss = - (torch.log(1e-6 + torch.sigmoid(pos_score - neg_score))).sum(-1)
        rec_loss = rec_loss / batch

        if self.use_quantizer:
            return rec_loss + self.quantizer_loss_weight * quantizer_loss

        return rec_loss
       
    @torch.no_grad()
    def predict(self, item_seq, item_feature):     
        item_seq = self.reconstruct_test_data(item_seq)
     
        input_items_embs = item_feature[item_seq] #[batch, max_seq_len+1, dim]
     
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_items_embs + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
      
        seq_output = output_embs[-1][:, -1]
        scores = torch.matmul(seq_output, item_feature[:self.item_num].t())  
        return scores

    # @torch.no_grad()
    # def compute_item(self, item):
    #     return self.visual_encoder(item)

    @torch.no_grad()
    def compute_item(self, item):
        if self.modal_fusion == 'txt':
            tokens = self.text_item_content[item]

            with torch.autocast(device_type='cuda', enabled=False):
                result = self.nlp_encoder(tokens)

            item_emb = F.normalize(result, dim=-1)
        elif self.modal_fusion == 'img':
            item_emb = self.visual_encoder(item)
        else:  # fused
            batch_size = item.shape[0]
            item_ids = torch.arange(
                self._compute_cursor,
                self._compute_cursor + batch_size,
                dtype=torch.long,
                device=item.device,
            )
            self._compute_cursor += batch_size
            
            # 走完一轮后重置
            if self._compute_cursor >= self.bert_item_features.shape[0]:
                self._compute_cursor = 0
            
            bert_emb = self.bert_item_features[item_ids]
            txt_emb = self.nlp_encoder.text_encoders['title'].activate(
                self.nlp_encoder.text_encoders['title'].fc(bert_emb)
            )
            txt_emb = F.normalize(txt_emb, dim=-1)
            img_emb = self.visual_encoder(item)
            item_emb = 0.5 * img_emb + 0.5 * txt_emb
        item_emb, _ = self.apply_quantizer_feature(item_emb)
        return item_emb

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
