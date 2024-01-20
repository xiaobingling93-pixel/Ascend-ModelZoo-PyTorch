# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import platform
import argparse

import numpy as np
import torch
import torch_aie
import onnxruntime as rt
from scipy.io.wavfile import write
from scipy.special import expit
from tacotron2.text import text_to_sequence
import time


def compute_fps_encoder(model_eval, seqs, seq_lens, warm):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0

    seqs = seqs.to("npu:0")
    seq_lens = seq_lens.to("npu:0")

    if (warm):
        encoder_output = model_eval(seqs, seq_lens)
        default_stream.synchronize()
    else:
        t0 = time.time()
        encoder_output = model_eval(seqs, seq_lens)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)
        print("encoder infer time".format(time_cost))

    seqs = seqs.cpu()
    seq_lens = seq_lens.cpu()
    seq_lens = seq_lens.numpy()

    return encoder_output, time_cost, seq_lens

def compute_fps_decoder(model_eval, decoder_inputs, warm):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0

    decoder_input_npu = []
    for i in range(11):
        decoder_input_npu.append(decoder_inputs[i].to("npu:0"))
    
    if warm:
        decoder_iter_output_npu = model_eval(*decoder_input_npu)
        default_stream.synchronize()
    else:
        t0 = time.time()
        decoder_iter_output_npu = model_eval(*decoder_input_npu)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)

    decoder_iter_output = []
    for i in range(len(decoder_iter_output_npu)):
        decoder_iter_output.append(decoder_iter_output_npu[i].cpu())
    return decoder_iter_output, time_cost
   
def compute_fps_posnet(model_eval, mel_outputs, warm):
    
    default_stream = torch_aie.npu.default_stream()   
    time_cost = 0

    mel_outputs = mel_outputs.to("npu:0")

    if warm:
        mel_outputs_postnet = model_eval(mel_outputs)
        default_stream.synchronize()
    else:
        t0 = time.time()
        mel_outputs_postnet = model_eval(mel_outputs)
        default_stream.synchronize()
        t1 = time.time()
        time_cost += (t1 - t0)

    mel_outputs_postnet = mel_outputs_postnet.cpu()
    return mel_outputs_postnet, time_cost


def pad_sequences(batch_seqs, batch_names):
    import copy
    batch_copy = copy.deepcopy(batch_seqs)
    for i in range(len(batch_copy)):
        if len(batch_copy[i]) > args.max_input_len:
            batch_seqs[i] = batch_seqs[i][:args.max_input_len]

    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch_seqs]), dim=0, descending=True)

    text_padded = torch.LongTensor(len(batch_seqs), input_lengths[0])
    text_padded.zero_()
    text_padded[0][:] += torch.IntTensor(text_to_sequence('.', ['english_cleaners'])[:])
    names_new = []
    for i in range(len(ids_sorted_decreasing)):
        text = batch_seqs[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text
        names_new.append(batch_names[ids_sorted_decreasing[i]])

    return text_padded, input_lengths, names_new


def prepare_input_sequence(batch_names, batch_texts):
    batch_seqs = []
    for i, text in enumerate(batch_texts):
        batch_seqs.append(torch.IntTensor(text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths, names_new = pad_sequences(batch_seqs, batch_names)

    text_padded = text_padded.long()
    input_lengths = input_lengths.long()

    return text_padded, input_lengths, names_new


def prepare_batch_wav(batch_size, wav_names, wav_texts, max_input):
    batch_texts = []
    batch_names = []
    for i in range(batch_size):
        if i == 0:
            batch_names.append(wav_names.pop(0))
            batch_texts.append(wav_texts.pop(0))
        else:
            batch_names.append(wav_names.pop())
            batch_texts.append(wav_texts.pop())
    if len(batch_texts[0]) < max_input:
        batch_texts[0] += ' a'
    return batch_names, batch_texts


def load_wav_texts(input_file):
    metadata_dict = {}
    if input_file.endswith('.csv'):
        with open(input_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                metadata_dict[line.strip().split('|')[0]] = line.strip().split('|')[-1]
    elif input_file.endswith('.txt'):
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                wav_name = line.split('|')[0].split('/', 2)[2].split('.')[0]
                wav_text = line.split('|')[1]
                metadata_dict[wav_name] = wav_text
    else:
        print("file is not support")

    wavs = sorted(metadata_dict.items(), key=lambda value: len(value[1]), reverse=True)
    wav_names = [wav[0] for wav in wavs]
    wav_texts = [wav[1].strip() for wav in wavs]

    return wav_names, wav_texts


def get_mask_from_lengths(lengths):
    lengths_tensor = torch.LongTensor(lengths)
    max_len = torch.max(lengths_tensor).item()
    ids = torch.arange(0, max_len, device=lengths_tensor.device, dtype=lengths_tensor.dtype)
    mask = (ids < lengths_tensor.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask



def sigmoid(inx):
    return expit(inx)

def update_decoder_inputs(decoder_inputs, decoder_outputs):
    new_decoder_inputs = [
        decoder_outputs[0],  # decoder_output
        decoder_outputs[2],  # attention_hidden
        decoder_outputs[3],  # attention_cell
        decoder_outputs[4],  # decoder_hidden
        decoder_outputs[5],  # decoder_cell
        decoder_outputs[6],  # attention_weights
        decoder_outputs[7],  # attention_weights_cum
        decoder_outputs[8],  # attention_context
        decoder_inputs[8],   # memory
        decoder_inputs[9],   # processed_memory
        decoder_inputs[10],  # mask
    ]
    return new_decoder_inputs


def inference_tacotron(seqs, seq_lens, encoder_model, decoder_model, posnet_model, total_time, warm):
    print("Starting run Tacotron2 encoder ……")
    encoder_output, time_cost, seq_lens = compute_fps_encoder(encoder_model, seqs, seq_lens, warm)
    total_time += time_cost

    mask_from_length_tensor = get_mask_from_lengths(seq_lens).numpy()
    mask_from_length_tensor = torch.from_numpy(mask_from_length_tensor)


    decoder_inputs = []
    decoder_inputs.append(torch.zeros((args.batch_size, 80), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, seq_lens[0]), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, seq_lens[0]), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 512), dtype=torch.float32))
    decoder_inputs.append(encoder_output[0])
    decoder_inputs.append(encoder_output[1])
    decoder_inputs.append(mask_from_length_tensor)


    gate_threshold = 0.5
    max_decoder_steps = 2000
    first_iter = True
    not_finished = torch.ones([args.batch_size], dtype=torch.int32)
    mel_lengths = torch.zeros([args.batch_size], dtype=torch.int32)

    print("Starting run Tacotron2 decoder ……")
    exec_seq = 0
    while True:

        exec_seq += 1

        
        decoder_iter_output, time_cost = compute_fps_decoder(decoder_model, decoder_inputs, warm)
        total_time += time_cost

        decoder_inputs = update_decoder_inputs(decoder_inputs, decoder_iter_output)

        if first_iter:
            mel_outputs = np.expand_dims(decoder_iter_output[0], 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate((mel_outputs, np.expand_dims(decoder_iter_output[0], 2)), 2)

        dec = torch.le(torch.Tensor(sigmoid(decoder_iter_output[1])), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if exec_seq > (seq_lens[0] * 6 + seq_lens[0] / 2):
            print("Warning! exec_seq > seq_lens, Stop after ", exec_seq, " decoder steps")
            break
        if mel_outputs.shape[2] == max_decoder_steps:
            print("Warning! Reach max decoder steps", max_decoder_steps)
            break
        if torch.sum(not_finished) == 0:
            print("Finished! Stop after ", mel_outputs.shape[2], " decoder steps")
            break

    print("Starting run Tacotron2 postnet ……")

    mel_outputs = torch.from_numpy(mel_outputs)

    mel_outputs_postnet, time_cost = compute_fps_posnet(posnet_model, mel_outputs, warm)
    total_time += time_cost

    print("Tacotron2 infer success")

    return total_time, mel_outputs_postnet



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', type=str, required=True)
    parser.add_argument('--decoder_model_path', type=str, required=True)
    parser.add_argument('--posnet_model_path', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input text')
    parser.add_argument('-o', '--output', required=False, default="output/audio_pt_dyn", type=str,
                       help='output folder to save autio')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--max_input_len', default=256, type=int, help='max input len') 
    parser.add_argument('--gen_wave', action='store_true', help='generate wav file')
    parser.add_argument('--stft_hop_length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')

    args = parser.parse_args()

    encoder_model = torch.jit.load(args.encoder_model_path)
    decoder_model = torch.jit.load(args.decoder_model_path)
    posnet_model = torch.jit.load(args.posnet_model_path)


    torch_aie.set_device(0)
    os.makedirs(args.output, exist_ok=True)

    # load wav_texts data
    wav_names, wav_texts = load_wav_texts(args.input)


    iter = 0
    total_time = 0
    all_mels = 0
    warm = True
    while args.batch_size <= len(wav_texts):
        if (iter == 5):
            break
        
        if (iter >= 2):
            warm = False
        
        # data preprocess (prepare batch & load)
        batch_names, batch_texts = prepare_batch_wav(args.batch_size, wav_names, wav_texts, args.max_input_len)
        seqs, seq_lens, batch_names_new = prepare_input_sequence(batch_names, batch_texts)
        if seqs == '':
            print("Invalid input!")
            break
        seqs = seqs.to(torch.int64).numpy()
        seq_lens = seq_lens.to(torch.int32).numpy()

        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)

        total_time, mel_outputs_postnet = inference_tacotron(seqs, seq_lens, encoder_model.eval(), decoder_model.eval(), posnet_model.eval(), total_time, warm)
        print("mel_outputs_postnet.shape[2]: {}".format(mel_outputs_postnet.shape[2]))

        num_mels = mel_outputs_postnet.shape[0] * mel_outputs_postnet.shape[2]

        if not warm:
            all_mels += num_mels
        
        iter+=1
    

    print(f" mel fps: {all_mels}  / {total_time : .3f} = {(all_mels / total_time):.3f} samples/s")