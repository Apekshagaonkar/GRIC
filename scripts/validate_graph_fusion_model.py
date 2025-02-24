from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import tqdm
import torch_geometric

import pickle
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Blip2Retreiver(nn.Module):
    def __init__(self, model_name, load_in_8bit=True, device_map=None, torch_dtype=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained( "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)

        self.mlp = nn.Sequential(nn.Linear(768, 1408), nn.ReLU(), nn.Linear(1408, 1408)).to(self.device, torch.float16)
        self.graph_conv1 = torch_geometric.nn.GCNConv(1408, 1408).to(self.device, torch.float16)


    def forward(self, pixel_values=None, input_ids=None, text_clip=None, scores = None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, interpolate_pos_encoding=False):
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        global_image_embeds = image_embeds[:, 0, :].reshape(-1, 1, 1408)

        k = text_clip.shape[1]
        #MLP
        text_clip = self.mlp(text_clip)

        #Graph Convolution
        x = torch.cat([global_image_embeds, text_clip], dim=1)
        edge_index = []
        for i in range(k):
            edge_index.append([0, i])
        for i in range(k):
            edge_index.append([i, 0])

        edge_index = np.array(edge_index)
        edge_indices = []
        for i in range(len(pixel_values)):
            edge_indices.append(edge_index + i*k)
        edge_index = np.concatenate(edge_indices)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T

        edge_attr = scores.repeat(1,2).to(self.device).flatten()
        x = self.graph_conv1(x.flatten(end_dim=-2), edge_index, edge_attr)
        x = torch.relu(x)

        x = x.reshape(-1, 1+k, 1408)

        image_embeds[:, 0, :] = x[:, 0, :]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.model.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.model.config.use_decoder_only_language_model:
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.model.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )
    
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, text_clip=None, scores = None, decoder_start_token_id=None, decoder_end_token_id=None, generate_kwargs=None, interpolate_pos_encoding=False):
        if hasattr(self.model, "hf_device_map"):
        # preprocess for `accelerate`
            self.model._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state

        global_image_embeds = image_embeds[:, 0, :].reshape(-1, 1, 1408)

        k = text_clip.shape[1]
        #MLP
        text_clip = self.mlp(text_clip)

        #Graph Convolution
        x = torch.cat([global_image_embeds, text_clip], dim=1)
        edge_index = []
        for i in range(k):
            edge_index.append([0, i])
        for i in range(k):
            edge_index.append([i, 0])

        edge_index = np.array(edge_index)
        edge_indices = []
        for i in range(batch_size):
            edge_indices.append(edge_index + i*k)
        edge_index = np.concatenate(edge_indices)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T

        edge_attr = scores.repeat(1,2).to(self.device).flatten()
        x = self.graph_conv1(x.flatten(end_dim=-2), edge_index, edge_attr)
        x = torch.relu(x)

        x = x.reshape(-1, 1+k, 1408)

        image_embeds[:, 0, :] = x[:, 0, :]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.model.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.model.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        # TODO (joao, raushan): refactor `generate` to avoid these operations with VLMs
        if not self.model.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.model.language_model.config.is_encoder_decoder:
            bos_tokens = (
                torch.LongTensor([[self.model.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)
        return outputs

class TrainDataset(Dataset):
    def __init__(self, k = 1, path='/3d_data/datasets/coco/', knn_file = 'knn/kNN.npy', dict_file ='image_name.pickle', image_2_cap = 'image_name_2_captions.pickle'):

        self.root = path
        self.kNN = np.load(self.root + knn_file, allow_pickle=True)

        with open(self.root + dict_file, 'rb') as handle:
            self.max_caption_dict = pickle.load(handle)

        with open(self.root + image_2_cap, 'rb') as handle:
            self.image_caption_dict = pickle.load(handle)

        self.img_names = sorted(list(self.max_caption_dict.keys()))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_names = [name.split('/')[1].split('.')[0] for name in self.img_names]
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

        captions = [self.image_caption_dict[name + '.jpg'][self.max_caption_dict['train_emb/'+name+'.npy']] for name in self.img_names]
        neighbor_captions = [self.image_caption_dict[name + '.jpg'][self.max_caption_dict['train_emb/'+name+'.npy']] + ' Rephrase' for name in self.img_names]

        self.caption_ids = self.processor.tokenizer(text = captions, return_tensors="pt", padding='max_length', truncation=True, max_length = 20).input_ids.to(self.device)
        self.neighbor_ids = self.processor.tokenizer(text = neighbor_captions, return_tensors="pt", padding='max_length', truncation=True, max_length = 20).input_ids.to(self.device)

        if os.path.exists(self.root + 'images.pt'):
            self.images = torch.load(self.root + 'images.pt')
        else:
            #load all im ages with batch size
            img_names_splits = [self.img_names[i:i + 256] for i in range(0, len(self.img_names), 256)]
            self.images = []
            for img_names in tqdm.tqdm(img_names_splits):
                images = []
                for img_name in img_names:
                    image = Image.open(self.root + 'train2014/' + img_name + '.jpg')
                    images.append(image)
                images = self.processor.image_processor(images=images, return_tensors="pt").to(self.device, torch.float16)
                self.images.append(images.pixel_values)

            self.images = torch.cat(self.images, dim=0)

            #save all images in a file
            torch.save(self.images, self.root + 'images.pt')

    def __getitem__(self, idx):
        #img embedding, caption embedding, kNN scores, kNN indices

        image = self.images[idx]
        scores, indices = self.kNN[idx]
        
        max_caption_ids = self.neighbor_ids[int(indices[0])]
        caption_ids = self.caption_ids[idx]
        attention_mask = torch.ones(max_caption_ids.shape).to(self.device)

        return max_caption_ids.to(self.device), image, attention_mask, caption_ids.to(self.device)

    def __len__(self):
        return len(self.img_names)
    
class ValidationDataset(Dataset):
    def __init__(self, k = 1, path='/3d_data/datasets/coco/', knn_file = 'knn/kNN_val.npy', dict_file ='image_name_val.pickle', image_2_cap = 'image_name_2_captions_val.pickle', train_dict_file = 'image_name.pickle', train_image_2_cap = 'image_name_2_captions.pickle'):

        self.root = path
        self.kNN = np.load(self.root + knn_file, allow_pickle=True)

        with open(self.root + dict_file, 'rb') as handle:
            self.max_caption_dict = pickle.load(handle)

        with open(self.root + image_2_cap, 'rb') as handle:
            self.image_caption_dict = pickle.load(handle)

        with open(self.root + train_dict_file, 'rb') as handle:
            self.train_max_caption_dict = pickle.load(handle)

        with open(self.root + train_image_2_cap, 'rb') as handle:
            self.train_image_caption_dict = pickle.load(handle)

        self.img_names = sorted(list(self.max_caption_dict.keys()))
        self.img_names = [name[8:].split('.')[0] for name in self.img_names]

        self.train_img_names = sorted(list(self.train_max_caption_dict.keys()))
        self.train_img_names = [name.split('/')[1].split('.')[0] for name in self.train_img_names]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

        captions = [self.image_caption_dict[name + '.jpg'][self.max_caption_dict["val_emb\\" +name+'.npy']] for name in self.img_names]
        train_captions = [self.train_image_caption_dict[name + '.jpg'][self.train_max_caption_dict['train_emb/'+name+'.npy']] for name in self.train_img_names]

        neighbor_captions = [  ', '.join([train_captions[j] for j in self.kNN[idx][1][:k]]) + ' Summarize' for idx, name in enumerate(self.img_names)]

        self.caption_ids = self.processor.tokenizer(text = captions, return_tensors="pt", padding='max_length', truncation=True, max_length = 20).input_ids.to(self.device)
        self.neighbor_ids = self.processor.tokenizer(text = neighbor_captions, return_tensors="pt", padding='max_length', truncation=True, max_length = 20).input_ids.to(self.device)

        if os.path.exists(self.root + 'val_images.pt'):
            self.images = torch.load(self.root + 'val_images.pt')
        else:
            #load all im ages with batch size
            img_names_splits = [self.img_names[i:i + 256] for i in range(0, len(self.img_names), 256)]
            self.images = torch.zeros((len(self.img_names), 3, 224, 224), device = self.device, dtype=torch.float16).contiguous()
            for idx, img_names in tqdm.tqdm(enumerate(img_names_splits), total=len(img_names_splits)):
                images = []
                for img_name in img_names:
                    image = Image.open(self.root + 'val2014/' + img_name + '.jpg')
                    images.append(image)
                images = self.processor.image_processor(images=images, return_tensors="pt").to(self.device, torch.float16)
                self.images[idx*256:idx*256+len(images.pixel_values)] = images.pixel_values

            #save all images in a file
            torch.save(self.images, self.root + 'val_images.pt')

        print(torch.all(self.images[100] == self.images[2]))

    def __getitem__(self, idx):
        #img embedding, caption embedding, kNN scores, kNN indices

        image = self.images[idx]
        scores, indices = self.kNN[idx]
        
        max_caption_ids = self.neighbor_ids[int(indices[0])]
        caption_ids = self.caption_ids[idx]
        attention_mask = torch.ones(max_caption_ids.shape).to(self.device)

        return max_caption_ids.to(self.device), image, attention_mask, caption_ids.to(self.device), idx

    def __len__(self):
        return len(self.img_names)
    

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    save_path = '/3d_data/retreiver/base/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)

    for param in model.language_model.parameters():
        param.requires_grad = False

    for param in model.vision_model.parameters():
        if type(param) == torch.nn.parameter.Parameter:
            param.requires_grad = False 

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)

    # train_data = TrainDataset()
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # model.train()

    # best_loss = 1000

    # print('==================== Training Started ====================')
    # for epoch in range(100):
    #     loss_avg = 0
    #     for i, (input_ids, pixel_values, attention_masks, caption_ids) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
    #         optimizer.zero_grad()

    #         outputs = model(pixel_values = pixel_values, input_ids = input_ids, attention_mask = attention_masks, labels=caption_ids)
    #         loss = outputs.loss
            
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()

    #         loss_avg += loss.item()

    #     loss_avg /= len(train_loader)

    #     if loss_avg < best_loss:
    #         best_loss = loss_avg
    #         torch.save(model.state_dict(), save_path + 'best_model.pt')
    #         print(f'Best Model Saved with Loss: {best_loss:.4f}')

    #     print(f'Epoch: {epoch+1}, Loss: {loss_avg:.4f}')

    model.eval()

    val_data = ValidationDataset()
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    predictions = {}

    for i, (input_ids, pixel_values, attention_masks, caption_ids, loc_ids) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
        output = model.generate(pixel_values = pixel_values)

        prediction_strs = val_data.processor.batch_decode(output, skip_special_tokens=True)

        for j in range(len(loc_ids)):
            predictions[val_data.img_names[loc_ids[j]]] = prediction_strs[j].strip()


    with open(save_path + 'predictions.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

