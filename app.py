
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

st.set_page_config(page_title="Visual Question Answering")
st.title("🖼️ Visual Question Answering")

@st.cache_resource
def load_model():
    p=BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    m=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return p,m

processor,model=load_model()

img=st.file_uploader("Upload an image",type=["png","jpg","jpeg"])
q=st.text_input("Ask a question about the image")

if img:
    image=Image.open(img).convert("RGB")
    st.image(image,caption="Uploaded Image",width='stretch')
    if st.button("Answer") and q:
        inputs=processor(image,q,return_tensors="pt")
        out=model.generate(**inputs)
        ans=processor.decode(out[0],skip_special_tokens=True)
        st.success(ans)