import streamlit as st
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "transformer_full_new.pth"
model = torch.load(model_path, map_location=device)
model.eval()

def generate_code(model, pseudocode, sp_pseudo, sp_code, max_len=500):
    model.eval()
    src_tokens = sp_pseudo.encode_as_ids(pseudocode)
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    tgt = torch.full((1, 1), 2, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(-1).item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == 5:
                break
    
    generated_tokens = tgt.squeeze(0).tolist()
    generated_code = sp_code.decode_ids(generated_tokens)
    return generated_code

# Streamlit UI
st.title("Pseudocode to C++ Converter")
st.write("Enter your pseudocode below and generate equivalent C++ code.")

pseudocode_input = st.text_area("Pseudocode Input", "")

if st.button("Generate C++ Code"):
    if pseudocode_input.strip():
        generated_output = generate_code(model, pseudocode_input, sp_pseudo, sp_code, max_len=500)
        st.subheader("Generated C++ Code:")
        st.code(generated_output, language='cpp')
    else:
        st.warning("Please enter some pseudocode.")
