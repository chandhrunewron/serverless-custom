from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
from potassium import Potassium, Request, Response

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():

    global model
    global tokenizer
    print("loading to CPU...")
    base_model = "TinyPixel/Llama-2-7B-bf16-sharded"
    tuned_adapter = "newronai/llama-2-7b-QLoRA-Trial1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    
    config = PeftConfig.from_pretrained(tuned_adapter)
    model = AutoModelForCausalLM.from_pretrained(base_model,quantization_config=bnb_config,use_cache = "cache",trust_remote_code=True,low_cpu_mem_usage=True).to("cuda:0")

    model = PeftModel.from_pretrained(model, tuned_adapter,use_cache="cache",low_cpu_mem_usage=True ).to("cuda:0")
    # context = {"model":model,"tokenizer":tokenizer}
    
    # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    # tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_cache="cache",trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model.generate(input_tokens)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
