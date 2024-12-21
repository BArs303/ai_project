import streamlit as st
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

st.title("AI translator")
text = st.text_input(label="abc")
#command = "summarize | "
#command = "translate en-ru | "
command = "translate ru-en | "

#st.write(command+text)

model_name = "cointegrated/rut5-base-multitask"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)
if text:    
	st.write(generate(command+text))
