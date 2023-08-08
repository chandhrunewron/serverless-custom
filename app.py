from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
from potassium import Potassium, Request, Response

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
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
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 use_cache = "cache",
                                                 low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, tuned_adapter)
    # context = {"model":model,"tokenizer":tokenizer}
    
    # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    # tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_cache="cache")
    tokenizer.pad_token = tokenizer.eos_token


@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_new_tokens = request.json.get("max_new_tokens")

    tokenizer = context.get("tokenizer")
    model = context.get("model")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=int(max_new_tokens))
    output = tokenizer.decode(outputs[0])

    return Response(
        json = {"outputs": output}, 
        status=200
    )
# Inference is ran for every server call
# Reference your preloaded global model variable here.
# def inference(model_inputs:dict) -> dict:
#     global model
#     global tokenizer
#     print("Inferring...")
#     # Parse out your arguments
#     prompt = model_inputs.get('prompt', None)
#     if prompt == None:
#         return {'message': "No prompt provided"}
#     print("Tokenizing inputs")
#     # Tokenize inputs
#     input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     # Run the model
#     output = model.generate(input_tokens)

#     # Decode output tokens
#     output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    # result = {"output": output_text}

    # # Return the results as a dictionary
    # return result
if __name__ == "__main__":
    app.serve()
