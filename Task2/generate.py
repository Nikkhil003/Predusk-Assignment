from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generation function
def generate_text(temperature):
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Generate with two temperatures
gen_temp_07 = generate_text(temperature=0.7)
gen_temp_10 = generate_text(temperature=1.0)

# Print outputs to console
print("\n=== Generation with Temperature 0.7 ===\n")
print(gen_temp_07)

print("\n=== Generation with Temperature 1.0 ===\n")
print(gen_temp_10)

# Save to file
with open("generated_outputs.txt", "w") as f:
    f.write("=== Generation with Temperature 0.7 ===\n")
    f.write(gen_temp_07 + "\n\n")
    f.write("=== Generation with Temperature 1.0 ===\n")
    f.write(gen_temp_10 + "\n")
