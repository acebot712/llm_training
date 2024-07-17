from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(model)

model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/abhijoy/model_compressor/models/Pintxo1.0-90-Llama2-7B")
print(model)
