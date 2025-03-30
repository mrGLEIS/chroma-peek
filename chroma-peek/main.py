import streamlit as st
import pandas as pd
from utils.peek import ChromaPeek

st.set_page_config(page_title="chroma-peek", page_icon="👀")

## styles ##
st.markdown(
    """ <style>
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
        </style> """, 
    unsafe_allow_html=True
)

st.title("Chroma Peek 👀")

# Get URI of the persist directory
path = ""
col1, col2 = st.columns([4, 1])  # Adjust the ratio as needed
with col1:
    path = st.text_input("Enter persist path", placeholder="Paste full path of persist")
with col2:
    st.write("")
    if st.button('🔄'):
        st.rerun()

st.divider()

# Load collections
if path:
    peeker = ChromaPeek(path)
    
    collections = peeker.get_collections()
    
    if collections:
        ## Create radio button for selecting a collection
        col1, col2 = st.columns([1, 3])
        with col1:
            collection_selected = st.radio(
                "Select collection to view",
                options=collections,
                index=0
            )
        
        with col2:
            df = peeker.get_collection_data(collection_selected, dataframe=True)

            st.markdown(f"<b>Data in </b>*{collection_selected}*", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=300)
            
        st.divider()

        query = st.text_input("Enter query to get 3 similar texts", placeholder="Get 3 similar texts")
        if query:
            result_df = peeker.query(query, collection_selected, dataframe=True)
            st.dataframe(result_df, use_container_width=True)
    else:
        st.warning("No collections found in the given directory.")
else:
    st.subheader("Enter a valid full persist path")
