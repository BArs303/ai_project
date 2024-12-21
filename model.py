import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
model_name = "cointegrated/rut5-base-multitask"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


command = "translate ru-en | "
#command = "summarize | "
text = "Самые лучшие истории придумывает жизнь" 
print(command + text)
print(generate(command+text))
