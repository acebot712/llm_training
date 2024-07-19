from transformers import AutoModelForCausalLM, AutoTokenizer
from model_loader import LlamaCausalLMTensor

# Load the models
# model_2 = LlamaCausalLMTensor.from_pretrained("/home/ubuntu/profiler/Downloaded_checkpoint_pintxo")
# model_1 = LlamaCausalLMTensor.from_pretrained("/home/ubuntu/profiler/compact/compact_Pintxo_999_1000_supra3_no_gate_train_zero")
model_3 = LlamaCausalLMTensor.from_pretrained("/home/ubuntu/profiler/compact/compact_Pintxo_1071_1000_supra3_no_gate_mpo_zero_98")
# model_4 = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

# print(model_4)

print(model_3)
# print(model_1)
# Function to print layer types
# def print_layer_types(model):
#     for name, module in model.named_modules():
#         print(f"{name}: {type(module)}")

# # print("Model 1 Layers:")
# # print_layer_types(model_1)

# for params in model_1.parameters():
#     # print(f"{params=}")
#     print(params.shape)



# # print("\nModel 2 Layers:")
# # print_layer_types(model_2)


# print number of parameters
print(f"Number of parameters in model 1: {sum(p.numel() for p in model_3.parameters())}")

# def main(prompt):
#     hf_model_path = "/home/ubuntu/profiler/compact/compact_Pintxo_999_1000_supra3_no_gate_train_zero"

#     pretrain_model = LlamaCausalLMTensor.from_pretrained(hf_model_path, local_files_only=True)
#     # print(f"Pretrained model: {pretrain_model}")
#     device = "cuda"
#     pretrain_model.to(device)  # Move model to GPU if available
#     print("Model loaded successfully.")
#     tokenizer = AutoTokenizer.from_pretrained(
#         hf_model_path, add_bos_token=False, add_eos_token=False
#     )

#     chat = [
#         {"role": "user", "content": prompt},
#     ]
#     prompt = tokenizer.apply_chat_template(chat, tokenize=False)

#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
#     inputs.to(device)  # Move input tensors to GPU if available

#     generated_text = ""
#     for _ in range(1):  # Limiting to 5 iterations for demonstration
#         sample_output = pretrain_model.generate(
#             **inputs, max_new_tokens=512, do_sample=True, top_p=0.92, top_k=0
#         )
#         decoded_output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
#         if "</s>" in decoded_output:
#             decoded_output = decoded_output.split("</s>")[0] + "</s>"
#             generated_text += decoded_output
#             break
#         generated_text += decoded_output
#         inputs["input_ids"] = sample_output  # Update input for next generation

#     print("Output:\n" + 100 * '-')
#     print(generated_text)


# if __name__ == "__main__":
#     prompt = "Where is spain? "
#     main(prompt)
