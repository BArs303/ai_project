import streamlit as st
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

st.title("AI translator")
text = st.text_input(label="Text input field")

command_type = st.radio(
    "Model type:",
    ["Translator", "paraphrase"],
)

if command_type == "Translator":
   command = "translate en-ru | "
else:
    command = "paraphrase | "


model_name = "cointegrated/rut5-base-multitask"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)

if text:   
    if command == "paraphrase | ":
        st.write(generate(command+text), encoder_no_repeat_ngram_size=1, repetition_penalty=0.5, no_repeat_ngram_size=1)
        st.write("paraphrase option selected")
    else:
        st.write(generate(command+text))

