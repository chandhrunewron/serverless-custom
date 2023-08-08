# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    global model
    global tokenizer

    base_model = "TinyPixel/Llama-2-7B-bf16-sharded"
    tuned_adapter = "newronai/llama-2-7b-QLoRA-Trial1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    
    config = PeftConfig.from_pretrained(tuned_adapter)
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 quantization_config=bnb_config, 
                                                 torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 use_cache = "cache")
    model = PeftModel.from_pretrained(model, tuned_adapter)
    
    # tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_cache = "cache")
    tokenizer.pad_token = tokenizer.eos_token



if __name__ == "__main__":
    download_model()
