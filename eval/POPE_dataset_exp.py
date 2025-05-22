import argparse
import torch
import json
from tqdm import tqdm
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch.utils.data import DataLoader
from torch import nn
from accelerate import Accelerator
from typing import List, Tuple, Union


# Llava imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import kornia
from lavis.models import load_model_and_preprocess
# Perturbation & diffusion utilities
# from imagePerturbation import ImagePerturbation
from vcd_utils.vcd_add_noise import add_diffusion_noise

from guided_weight import *
from typing import Tuple
import numpy as np

from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()




def eval_models(args):

    disable_torch_init()
    model_path1 = os.path.expanduser(args.model_path1)
    model_path2 = os.path.expanduser(args.model_path2)

    model_name1 = get_model_name_from_path(model_path1)
    model_name2 = get_model_name_from_path(model_path2)


    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    tokenizer1, model1, image_processor1, context_len1 = load_pretrained_model(
    model_path1, args.model_base, model_name1, device = device1)

    model2, vis_processors2, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device2)
    models, tokenizers, processors = [], [], []
    print("model2", dir(model2))

    tokenizer2, model3, image_processor2, context_len2 = load_pretrained_model(
    model_path2, args.model_base, model_name2, device = device3)

    # llava_gen_cfg
    llava_gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=10,
        output_attentions = True,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True
    )

    # for blip
    blip2_gen_params = {
        "use_nucleus_sampling": True,
        "num_beams": 1,
        "top_p": args.top_p,
        "repetition_penalty": 1,
        "output_hidden_states": True,
        "output_scores": True,
        "output_attentions": True,
        "return_dict_in_generate": True
    }

   
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")


    for line in tqdm(questions):
        output_ans =[]
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        prompt2 = qs +  " Please answer this question with one word."
        if model1.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids1 = tokenizer_image_token(prompt, tokenizer1, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device1)
        attention_mask1 = torch.ones_like(input_ids1, dtype=torch.long).to(device1)

        input_ids2 = tokenizer_image_token(prompt, tokenizer2, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device2)
    
        # load images
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor1 = image_processor1.preprocess(image, return_tensors='pt')['pixel_values'][0]
        raw_image = image.convert("RGB")
        image_tensor2 = vis_processors2["eval"](raw_image).unsqueeze(0).to(device2)
        image_tensor3 = image_processor2.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #use image perturbation
        if args.use_cd:
            image_tensor_cd1 = add_diffusion_noise(image_tensor1, args.noise_step)
            image_tensor_cd2 = add_diffusion_noise(image_tensor2, args.noise_step)
            image_tensor_cd3 = add_diffusion_noise(image_tensor3, args.noise_step)
        else:
            image_tensor_cd1 = None
            image_tensor_cd2 = None
            image_tensor_cd3 = None


        input_length = [len(qs) for qs in input_ids1]

        distribution1, distribution2 = [], []
        max_length = args.max_new_tokens

        for i in range(max_length):
            if i == 0: #first step
                output1 = model1.generate(
                    input_ids = input_ids1,
                    images = image_tensor1.unsqueeze(0).half().to(device1),
                    images_cd =(image_tensor_cd1.unsqueeze(0).half().to(device1) if image_tensor_cd1 is not None else None),
                    # attention_mask = attention_mask1,
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    generation_config=llava_gen_cfg
                )  

                output2 = model2.generate(
                    {"image": image_tensor2, "prompt": prompt2},
                    images_cd=(image_tensor_cd2.half().to(device2) if image_tensor_cd2 is not None else None),
                    cd_beta=args.cd_beta,
                    cd_alpha=args.cd_alpha,
                    **blip2_gen_params
                )

                output3 = model3.generate(
                    input_ids = input_ids2,
                    images = image_tensor3.unsqueeze(0).half().to(device3),
                    images_cd =(image_tensor_cd3.unsqueeze(0).half().to(device3) if image_tensor_cd3 is not None else None),
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    generation_config=llava_gen_cfg
                )

            else:
                output1 = model1.generate(
                    input_ids = input_ids1,
                    images = image_tensor1.unsqueeze(0).half().to(device1),
                    images_cd =(image_tensor_cd1.unsqueeze(0).half().to(device1) if image_tensor_cd1 is not None else None),
                    # attention_mask = attention_mask1,
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    generation_config=llava_gen_cfg
                )  

                output2 = model2.generate(
                    {"image": image_tensor2, "prompt": prompt2},
                    images_cd=(image_tensor_cd2.half().to(device2) if image_tensor_cd2 is not None else None),
                    cd_beta=args.cd_beta,
                    cd_alpha=args.cd_alpha,
                    **blip2_gen_params
                )

                output3 = model3.generate(
                    input_ids = input_ids2,
                    images = image_tensor3.unsqueeze(0).half().to(device3),
                    images_cd =(image_tensor_cd3.unsqueeze(0).half().to(device3) if image_tensor_cd3 is not None else None),
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    generation_config=llava_gen_cfg
                )

            att = output1.attentions

            if args.use_cd:
                raw1 = output1.scores[0]["scores_cd"] 
                # model2
                raw2 = output2.scores[0]['scores_cd']
                raw3 = output3.scores[0]['scores_cd']
                logits1_l = output1.scores[0]['logits']
                logits2_l = output2.scores[0]['logits']

            else:
                raw1 = output1.scores[0]['logits']
                raw2 = output2.scores[0]['logits']
                raw3 = output3.scores[0]['logits']


            attn1 = output1.attentions[-1]
            attn2 = output2.attentions[-1]
            attn3 = output3.attentions[-1]

            logits1 = nn.functional.softmax(raw1, dim=-1).float().cpu()
            logits2 = nn.functional.softmax(raw2, dim=-1).float().cpu()
            logits3 = nn.functional.softmax(raw3, dim=-1).float().cpu()


            current_size = logits1.size(-1)
            logits2 = logits2[:,:current_size]
            logits_list = [logits1,
             logits2, logits3]
            attentions_list = [attn1, attn2, attn3]
            if args.fuse == "attention":

                ensembled_logits, model_weights = dynamic_ensemble_models(
                    logits_list,
                    attentions_list,
                    top_k_layer=3,
                    top_h_head=4,
                    lambda_step=0.05
                )
            if args.fuse == "uncertainty":
                ensembled_logits, model_weights = dynamic_ensemble_with_perplexity(
                    logits_list,
                    lambda_step=0.05
                )

            sample_greedy = True
            if sample_greedy: # greedy
                next_token_id = torch.argmax(ensembled_logits, dim=-1).item()
            else: # sampling
                cd_probs = nn.functional.softmax(ensembled_logits, dim=-1)
                next_token_id = torch.argmax(cd_probs, dim=-1).item()



            i1, m1 = [], []
            for input1_ids, mask1 in zip(input_ids1, attention_mask1):
                input1_ids = input1_ids.tolist()
                mask1 = mask1.tolist()
                input1_ids.append(next_token_id)
                mask1.append(1)
                i1.append(input1_ids)
                m1.append(mask1)


            input_ids1_new = torch.tensor(i1).to(device1)
            # attention_mask1 = torch.tensor(m1).to(device1)

            input_token_len = input_ids1.shape[1]
            outputs = tokenizer1.batch_decode(input_ids1_new[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            print("outputs",outputs)

        # decoding output
        for qs_len, ans in zip(input_length, input_ids1_new):
            output = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            output = ' ' .join(output.split())
            output_ans.append(output)
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": 'instruct_blip and llava',
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str,
                        default="Your data path")
    parser.add_argument("--prompts", type=str,
                        default="Your prompt path")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_path1", type=str, default="your model path")
    parser.add_argument("--model_path2", type=str, default="your model path")
    parser.add_argument("--output_file", type=str,
                        default="Your output file pathz")
    parser.add_argument('--image-folder', type=str, default='./MSCOCO/val2014')
    parser.add_argument('--question-file', type=str, default='./POPE/coco/coco_pope_random.json')
    parser.add_argument('--answers-file', type=str, default='./answer_ensemble.jsonl')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    # max_new_tokens for pope=1
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--use-cd', action='store_true', default = True)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument('--noise-step', type=int, default=999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fuse', type=str, default='uncertainty', choices=['uncertainty', 'attention'], help='ensemble method')
    args = parser.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()

    eval_models(args)


