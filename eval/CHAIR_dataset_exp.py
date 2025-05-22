import argparse
import torch
import json
from tqdm import tqdm
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import set_seed
from torch.utils.data import DataLoader
from torch import nn
from accelerate import Accelerator

# Llava imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import kornia
from lavis.models import load_model_and_preprocess
# Perturbation & diffusion utilities
from imagePerturbation import ImagePerturbation
from vcd_utils.vcd_add_noise import add_diffusion_noise

from guided_weight import *
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

import random
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
    "llava-next": "USER: <ImageHere> <question> ASSISTANT:"
}



def ensembled_token(logits1: torch.Tensor,
                    logits2: torch.Tensor):
    """
    logits1, logits2: [1, V]
    sample_fn: (probs: Tensor[V]) -> token_id: int
    """
    # 1) Calculate probabilities
    p1 = nn.functional.softmax(logits1, dim=-1).float().cpu()
    p2 = nn.functional.softmax(logits2, dim=-1).float().cpu()  # [1, V]

    # 2) Confidence
    conf1 = p1.max(dim=-1, keepdim=True).values  # [1,1]
    conf2 = p2.max(dim=-1, keepdim=True).values  # [1,1]

    # 3) Dynamic weight λ
    lam = conf1 / (conf1 + conf2 + 1e-8)         # [1,1]

    # 4) Ensemble
    p_ens = lam * p1 + (1 - lam) * p2            # [1, V]

    return p_ens


def boost_eos_prob(logits, eos_token_id, min_tokens=50, max_tokens=256, current_step=0):
    """
    Dynamically boost the probability of the EOS token based on generation length.

    Args:
        logits (torch.Tensor): The model output logits of shape (batch_size, vocab_size).
        eos_token_id (int): The token ID corresponding to the EOS (end-of-sequence) token.
        min_tokens (int): The minimum number of tokens before allowing EOS.
        max_tokens (int): The token length after which EOS is strongly encouraged.
        current_step (int): The current decoding step.

    Returns:
        torch.Tensor: Modified logits with adjusted EOS token probability.
    """
    # Clone logits to avoid modifying the original tensor
    logits = logits.clone()

    # Initialize boost factor
    boost_factor = 1.0

    if current_step < min_tokens:
        # Suppress EOS token before minimum length
        boost_factor = 0.01
    elif min_tokens <= current_step < max_tokens:
        # Linearly increase boost factor between min_tokens and max_tokens
        progress = (current_step - min_tokens) / (max_tokens - min_tokens)
        boost_factor = 1.0 + progress * 1.5  # Linearly scales from 1.0 to 2.5
    else:
        # After max_tokens, encourage EOS more strongly
        boost_factor = 3.0

    # Apply boost to EOS token probability
    logits[:, eos_token_id] *= boost_factor

    return logits




def eval_models(args):

    disable_torch_init()

    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model_path1 = os.path.expanduser(args.model_path1)
    model_path2 = os.path.expanduser(args.model_path2)
    model_path3 = os.path.expanduser(args.model_path3)

    model_name1 = get_model_name_from_path(model_path1)
    model_name3 = get_model_name_from_path(model_path3)

    # 1. Load LLaVA
    tokenizer1, model1, image_processor1, context_len1 = load_pretrained_model(
    model_path1, args.model_base, model_name1, device_map={"": device1}, device = device1)
    # 2. Load InstructBLIP and processor
    model2 = InstructBlipForConditionalGeneration.from_pretrained(
        model_path2,
        torch_dtype=torch.float16
        ).to(device2)
    processor2 = InstructBlipProcessor.from_pretrained(model_path2)

    tokenizer3, model3, image_processor3, context_len3 = load_pretrained_model(
    model_path3, args.model_base, model_name3, device_map={"": device3}, device = device3)

    img_files = os.listdir(args.image_folder)
    random.shuffle(img_files)

    with open('REPLACE BY YOUR ANNOTATIONS FILE PATH', 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_dict = {int(c["id"]): c["name"] for c in categories}
    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )

    # Set your output file path
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    file_path = 'REPLACE BY YOUR IMAGE PATH'
    image_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            digits = ''.join(c for c in line if c.isdigit())
            if digits:
                image_ids.append(int(digits))

    for img_id in range(len(img_files)):
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])

        if img_id not in image_ids:
            continue
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_save = {}
        img_save["image_id"] = img_id
        image_path = os.path.join(args.image_folder, img_file)
        image = Image.open(image_path)
        # Process image
        image_tensor1 = image_processor1.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device1)
        image_tensor3 = image_processor3.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device3)

        raw_image = image.convert("RGB")
        image_tensor2 = processor2(images=raw_image, return_tensors="pt").to(device2)
        pixel_values = image_tensor2.pixel_values

        # Prepare question
        qs = "Please describe this image in detail."

        template1 = INSTRUCTION_TEMPLATE["llava-1.5"]
        prompt1 = template1.replace("<question>", qs)

        input_ids1 = tokenizer_image_token(prompt1, tokenizer1, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device1)
        attention_mask1 = torch.ones_like(input_ids1, dtype=torch.long).to(device1)

        template2 = INSTRUCTION_TEMPLATE["instructblip"]
        prompt2 = template2.replace("<question>", qs)
        text_inputs = processor2(text=prompt2, return_tensors="pt").to(device2)
        input_ids2 = text_inputs.input_ids
        attention_mask2 = text_inputs.attention_mask

        template3 = INSTRUCTION_TEMPLATE["llava-next"]
        prompt3 = template3.replace("<question>", qs)
        input_ids3 = tokenizer_image_token(prompt3, tokenizer3, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device3)
        attention_mask3 = torch.ones_like(input_ids3, dtype=torch.long).to(device3)
        if args.use_cd:
            image_tensor_cd1 = add_diffusion_noise(image_tensor1, args.noise_step)
        else:
            image_tensor_cd1 = None

        eos_token_id1 = tokenizer1.eos_token_id

        eos_token_id2 = processor2.tokenizer.eos_token_id

        eos_token_id3 = tokenizer3.eos_token_id

        input_length = [len(qs) for qs in input_ids1] # record original input length
        prompt_token_len = input_length[0]

        max_length = args.max_new_tokens
        with torch.inference_mode():
            for i in range(max_length):
                if i == 0:
                    output1 = model1(
                        input_ids = input_ids1.to(device1),
                        images = image_tensor1.unsqueeze(0).half().to(device1),
                        output_attentions =True,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict = True,
                    )

                    output1_cd  = model1(
                        input_ids = input_ids1,
                        images =(image_tensor_cd1.unsqueeze(0).half().to(device1) if image_tensor_cd1 is not None else None),
                        output_attentions =True,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict = True,
                    )

                    output2 = model2(
                        input_ids=input_ids2,
                        pixel_values=pixel_values.half(),
                        qformer_input_ids = text_inputs.qformer_input_ids,
                        return_dict = True,
                        output_attentions = True,
                        output_hidden_states = True,
                    )

                    output3 = model3(
                        input_ids = input_ids3,
                        images = image_tensor3.unsqueeze(0).half().to(device3),
                        output_attentions =True,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict = True,
                    )

                else:
                    output1 = model1(
                        input_ids = input_ids1,
                        images = image_tensor1.unsqueeze(0).half().to(device1),
                        output_attentions =False,
                        output_hidden_states=False,
                        past_key_values =past_key_values1,
                        use_cache=True,
                        return_dict = True,
                    )
                    
                    output1_cd  = model1(
                        input_ids = input_ids1,
                        images =(image_tensor_cd1.unsqueeze(0).half().to(device1) if image_tensor_cd1 is not None else None),
                        output_attentions =True,
                        output_hidden_states=True,
                        past_key_values =past_key_values1_cd,
                        use_cache=True,
                        return_dict = True,
                    )

                    output2 = model2(
                        input_ids=input_ids2,
                        pixel_values=pixel_values.half(),
                        qformer_input_ids = text_inputs.qformer_input_ids,
                        return_dict=True,
                        output_attentions = False,
                        output_hidden_states = False,
                    )

                    output3 = model3(
                        input_ids = input_ids3,
                        images = image_tensor3.unsqueeze(0).half().to(device3),
                        output_attentions = False,
                        output_hidden_states = False,
                        use_cache=True,
                        return_dict = True,
                    )

                past_key_values1 = output1.past_key_values
                past_key_values1_cd =output1_cd.past_key_values

                if args.use_cd:
                    logits1 = (1+args.cd_alpha)* output1.logits[:, -1, :]- args.cd_alpha * output1_cd.logits[:, -1, :]
                    logits2 = output2.language_model_outputs["logits"][:, -1, :]
                    logits3 = output3.logits[:, -1, :]
                else:
                    logits1 = output1.logits[:, -1, :]
                    logits2 = output2.language_model_outputs["logits"][:, -1, :]
                    logits3 = output3.logits[:, -1, :]

               

                logits1 = boost_eos_prob(logits1, eos_token_id1, min_tokens=args.min_tokens, max_tokens= args.max_tokens, current_step=i)
                logits2 = boost_eos_prob(logits2, eos_token_id2, min_tokens=args.min_tokens, max_tokens= args.max_tokens, current_step=i)
                logits3 = boost_eos_prob(logits3, eos_token_id3, min_tokens=args.min_tokens, max_tokens= args.max_tokens, current_step=i)

                logits1 = nn.functional.softmax(logits1, dim=-1).float().cpu()
                logits2 = nn.functional.softmax(logits2, dim=-1).float().cpu()
                logits3 = nn.functional.softmax(logits3, dim=-1).float().cpu()

                current_size = logits1.size(-1)
                logits2 = logits2[:,:current_size]
                logits_list = [logits1, logits2,logits3]

                if args.fuse == "perplexity":
                    ensembled_logits, model_weights = dynamic_ensemble_with_perplexity(
                        logits_list,
                        lambda_step=0.05
                    )


                sample_greedy = True
                if sample_greedy: # greedy
                    next_token_id = torch.argmax(ensembled_logits, dim=-1)
                else:
                    next_token_id = torch.multinomial(ensembled_logits, num_samples=1)

                i1, m1 = [], []
                for input1_ids, mask1 in zip(input_ids1, attention_mask1):
                    input1_ids = input1_ids.tolist()
                    mask1 = mask1.tolist()
                    input1_ids.append(next_token_id)
                    mask1.append(1)
                    i1.append(input1_ids)
                    m1.append(mask1)

                #  Update input_ids1
                input_ids1_new = torch.tensor(i1).to(device1)
                input_ids1 = input_ids1_new
                attention_mask1 = torch.tensor(m1).to(device1)

                i2, m2 = [], []
                for input2_ids, mask2 in zip(input_ids2, attention_mask2):
                    input2_ids = input2_ids.tolist()
                    mask2 = mask2.tolist()
                    input2_ids.append(next_token_id)
                    mask2.append(1)
                    i2.append(input2_ids)
                    m2.append(mask2)

                # Update input_ids2
                input_ids2_new = torch.tensor(i2).to(device2)
                input_ids2 = input_ids2_new
                attention_mask2 = torch.tensor(m2).to(device2)

                i3, m3 = [], []
                for input3_ids, mask3 in zip(input_ids3, attention_mask3):
                    input3_ids = input3_ids.tolist()
                    mask3 = mask3.tolist()
                    input3_ids.append(next_token_id)
                    mask3.append(1)
                    i3.append(input3_ids)
                    m3.append(mask3)

                # Update input_ids3
                input_ids3_new = torch.tensor(i3).to(device3)
                input_ids3 = input_ids3_new
                attention_mask3 = torch.tensor(m3).to(device3)
                torch.cuda.empty_cache()

                input_token_len = input_ids1.shape[1]

        # Get the full generated ids
        full_generated_ids = input_ids1[0]

        # Find the first occurrence of the EOS token
        generated_part_ids = full_generated_ids[prompt_token_len:]
        eos_positions = []
        for eos_id in [eos_token_id1, eos_token_id2, eos_token_id3]:  # Check all models' EOS tokens
            positions = torch.where(generated_part_ids == eos_id)[0]
            if len(positions) > 0:
                eos_positions.append(positions[0].item())

        if eos_positions:
            first_eos_pos = min(eos_positions)
            output_ids_to_decode = generated_part_ids[:first_eos_pos]
            print(f"Find EOS token at {first_eos_pos}.")
        else:
            output_ids_to_decode = generated_part_ids
            print("No EOS token found, using all generated tokens.")

         # Decode the output ids
        output = tokenizer1.decode(output_ids_to_decode, skip_special_tokens=True)

        # Remove unwanted parts
        output = output.split("USER:")[0].strip()
        output = output.split("ASSISTANT:")[0].strip()
        output = output.split("<ImageHere>")[0].strip()
        output = ' '.join(output.split())

        img_save["caption"] = output
        ans_file.write(json.dumps(img_save)+ "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # === Dataset & Prompt ===
    parser.add_argument("--test_set", type=str, default="YOUR_TEST_SET_PATH", help="Path to test dataset")
    parser.add_argument("--prompts", type=str, default="YOUR_PROMPT_FILE_PATH", help="Prompt file for evaluation")

    # === Model Paths ===
    parser.add_argument("--model-base", type=str, default=None, help="Base model name if using LoRA or similar setup")
    parser.add_argument("--model_path1", type=str, default="YOUR_MODEL_PATH_1", help="Path to first model (e.g. LLaVA-1.5)")
    parser.add_argument("--model_path2", type=str, default="YOUR_MODEL_PATH_2", help="Path to second model (e.g. InstructBLIP)")
    parser.add_argument("--model_path3", type=str, default="YOUR_MODEL_PATH_3", help="Path to third model (e.g. LLaVA-1.6)")

    # === Output ===
    parser.add_argument("--output_file", type=str, default="YOUR_OUTPUT_FILE_PATH", help="Path to save raw output")
    parser.add_argument('--answers-file', type=str, default='./answer_ensemble_perplexity_300.jsonl', help="Final answers output file")

    # === Image Input ===
    parser.add_argument('--image-folder', type=str, default='YOUR_IMAGE_FOLDER_PATH', help="Path to image dataset folder")

    # === Generation Setup ===
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    parser.add_argument("--num-chunks", type=int, default=1, help="For parallel inference: total number of chunks")
    parser.add_argument("--chunk-idx", type=int, default=0, help="For parallel inference: current chunk index")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum new tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")

    # === Contrastive Decoding Parameters ===
    parser.add_argument('--use-cd', action='store_true', default=True, help="Enable contrastive decoding")
    parser.add_argument("--cd_alpha", type=float, default=1.0, help="Contrastive decoding alpha")
    parser.add_argument("--cd_beta", type=float, default=0.1, help="Contrastive decoding beta")
    parser.add_argument('--noise-step', type=int, default=500, help="Diffusion noise step for CD")
    parser.add_argument('--min_tokens', type=int, default=50, help="Minimum tokens before allowing EOS")
    parser.add_argument('--max_tokens', type=int, default=256, help="Maximum tokens after which EOS is strongly encouraged")

    # === Misc ===
    parser.add_argument('--seed', type=int, default=52, help="Random seed")
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help="Penalty for repeating tokens. 1.0 means no penalty")
    parser.add_argument('--fuse', type=str, default='uncertainty', choices=['uncertainty', 'attention'], help='ensemble method')

    args = parser.parse_args()
    set_seed(args.seed)

    eval_models(args)

