import streamlit as st
import pandas as pd 
import random
import os
from groq import Groq
import plotly.graph_objects as go
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chains import LLMChain

df_cv_score = pd.read_csv('final_cv_scores.csv')
Dataset = pd.read_csv('final_ranked_resumes.csv')
df_tags = pd.read_csv('final_processed_resumes.csv')


st.set_page_config(layout="wide")

# Define custom CSS
css = """
<style>
    /* Background color of the main content area */
    .stApp {
        background: #000000
        # ; /* Light blue to white gradient */
    }

    /* Sidebar color */
    .css-1d391kg {
        background: #000000
        # linear-gradient(to bottom, #ADD8E6, #FFFFFF); /* Light blue to white gradient */
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Display the main title
st.markdown(
    """
    <h1 style='font-size: 80px; text-align: center;'>Resume Scorer🔎</h1>
    """, 
    unsafe_allow_html=True
)

# Display the candidate database title
st.markdown(
    """
    <hr>
    <h1>Candidate Database</h1>
    """, 
    unsafe_allow_html=True
)
st.markdown(f"<hr>", unsafe_allow_html=True)
# Initialize session state variables
if 'view_profile' not in st.session_state:
    st.session_state.view_profile = False
if 'current_profile_id' not in st.session_state:
    st.session_state.current_profile_id = None
if 'offset' not in st.session_state:
    st.session_state.offset = 0
if 'limit' not in st.session_state:
    st.session_state.limit = 10  # Number of candidates to display at a time
if 'displayed_candidates' not in st.session_state:
    st.session_state.displayed_candidates = []  # Store the IDs of candidates to display
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'active_tags' not in st.session_state:
    st.session_state.active_tags = []

# Define available tags
available_tags = ['Accounting', 'administrative', 'budgets', 'documentation', 'financial', 
                  'coaching', 'Excel', 'hardware', 'delivery', 'banking', 'inventory']

       
st.subheader("Skill Filter")
selected_tags = st.multiselect("Select Active Tags", available_tags, default=st.session_state.active_tags)

st.session_state.active_tags = selected_tags

df_tags['Keywords'] = df_tags['Keywords'].apply(eval)

if not st.session_state.active_tags:
    df_filtered = Dataset # No filter applied
else:
    matching_indices = [
        index for index, row in df_tags.iterrows()
        if any(active_tag in row['Resume_Skills'] for active_tag in st.session_state.active_tags)
    ]
    
    df_filtered = Dataset.loc[matching_indices]  

# Define columns to display
columns_to_display = ['ID', 'Education_Level', 'Years_of_Experience', 'Overall_Score']

# Check if any filters are applied
if not st.session_state.active_tags:  # No filters, display the full dataset
    Dataset_sorted = Dataset.sort_values(by='Overall_Score', ascending=False)
    st.write("Displaying the full dataset:")
else:  # Filters are applied, display the filtered dataset
    Dataset_sorted = df_filtered.sort_values(by='Overall_Score', ascending=False)
    st.write("Displaying the filtered dataset:")

# Calculate the total number of candidates to display
total_candidates_to_display = st.session_state.offset + st.session_state.limit
total_candidates = len(Dataset_sorted)

# Update the displayed candidates list if necessary
if total_candidates_to_display > len(st.session_state.displayed_candidates):
    additional_candidates = Dataset_sorted.index[st.session_state.offset:total_candidates_to_display].tolist()
    st.session_state.displayed_candidates.extend(additional_candidates)

# Display header row
if not st.session_state.view_profile:
    st.markdown(f"<hr> ", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        st.markdown(f"<p style='font-size: 30px;'><b>ID</b></p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='font-size: 30px;'><b>Education</b></p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p style='font-size: 30px;'><b>Experience</b></p>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<p style='font-size: 30px;'><b>CV Score</b></p>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<p style='font-size: 30px;'><b>View Profile</b></p>", unsafe_allow_html=True)
    st.markdown(f"<hr> ", unsafe_allow_html=True)

    # Display each candidate in the displayed_candidates list
    for i in range(len(st.session_state.displayed_candidates)):
        candidate = Dataset_sorted.iloc[i]  # Use iloc to get the row by position
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  # Adjust column width ratio as needed

        with col1:
            st.markdown(f"<p style='font-size: 30px;'>{candidate['ID']}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='font-size: 30px;'>{candidate['Education_Level']}</p>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<p style='font-size: 30px;'>{candidate['Years_of_Experience']}</p>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<p style='font-size: 30px;'><b>{candidate['Overall_Score']:.2f}</b></p>", unsafe_allow_html=True)
        with col5:
            if st.button(f"View Profile for ID {candidate['ID']}"):
                st.session_state.view_profile = True
                st.session_state.current_profile_id = candidate['ID']

    if total_candidates_to_display < total_candidates:
        if st.button("Show More"):
            st.session_state.offset += st.session_state.limit  # Increment the offset
            # st.experimental_rerun()  # Rerun the app to display more candidates
    else:
        st.markdown("<p style='font-size: 30px;'>Showing all candidates.</p>", unsafe_allow_html=True)

else:
    # Function to generate a resume summary
    def generate_resume_summary(resume_text):
        client = Groq(api_key='your_api_key')

        prompt = f"Here is a resume text:\n{resume_text}\nGenerate a short summary with strengths and weaknesses of the candidate pointwise. Bold at appropriate words, 3 short points."
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content.strip()

    # Function to calculate sentiment based on provided text
    def sentiment_calculator(text_path):
        with open(text_path, 'r', encoding='Windows-1252') as f:
            text = f.read()

        # Initialize Groq client
        client = Groq(api_key=groq_api_key)

        # Create the completion request for sentiment analysis
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are provided a piece of text with various claims. Identify normal and exaggerated claims, assign sentiment scores, and output the result of:
                    (Average of normal sentiment scores) - 0.1 * (Average of exaggerated sentiment scores).
                    """
                },
                {
                    "role": "user",
                    "content": f"The text is provided below:\n{text}"
                }
            ],
            temperature=0.5,
            max_tokens=4096
        )
        return float(completion.choices[0].message.content.strip())

    # Initialize the Groq client
    groq_chat = Groq(api_key='your_api_key')

    # Function to generate a chat response directly with added context
    def generate_chat_response(user_question, candidate_id, resume_summary):
        # Add context about the candidate to guide the model
        prompt = f"""
        You are a helpful assistant with information about a candidate.
        Candidate ID: {candidate_id}.
        Summary of the candidate's resume: {resume_summary}.
        Now, answer the following question about this candidate:
        
        {user_question}
        """

        completion = groq_chat.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()


    # Profile and Chatbot View
    profile_id = st.session_state.get('current_profile_id', None)
    if profile_id:
        st.markdown(f"<h3 style='font-size: 30px;'>Displaying profile for ID: {profile_id}</h3>", unsafe_allow_html=True)
        
        # Display resume link and summary
        pdf_link = f"/Final_Resumes_Text/Resume_of_ID_{profile_id}.pdf"  
        st.markdown(f"[Resume of ID {profile_id}]({pdf_link})", unsafe_allow_html=True)

        text_data = pd.read_csv('resume_text.csv')
        if 'Text' in text_data.columns:
            resume_text = text_data["Text"].get(profile_id, "Resume text not available")

        # Generate and display resume summary
        try:
            summary = generate_resume_summary(f"Summarize the resume: {resume_text}")
            st.markdown(f"<hr><h3>Resume Summary</h3><hr><p>{summary}</p><hr>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating resume summary: {str(e)}")

        # Chatbot interaction
        st.title("Ask questions about the candidate")
        st.write("Hello! I'm your friendly Chatbot. Ask me anything about the candidate!")
        
        # Initialize chat history if not present
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Capture user question
        user_question = st.text_input("Ask a question:")

        # Respond to user question
        if user_question:
            # Generate response and update chat history
            summary = generate_resume_summary(f"Summarize the resume: {resume_text}")
            response = generate_chat_response(user_question,profile_id,summary)
            st.session_state.chat_history.append({'human': user_question, 'AI': response})
            st.write("Chatbot:", response)

        # Display chat history
        for msg in st.session_state.chat_history:
            st.write(f"**User:** {msg['human']}")
            st.write(f"**Chatbot:** {msg['AI']}")

        # Back button to reset profile view
        if st.button("Back to Candidate List"):
            st.session_state.view_profile = False
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.current_profile_id = None
            # st.experimental_rerun()

# sidebar content
# sidebar content
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 80px;'>ATS</h1>
        <div style='display: flex; align-items: center; justify-content: center;'>
        </div>
    </div>
    <hr>
    <p style="font-size: 20px; text-align: center;"><b>Rohan Chaudhary</b></p>
    
    <hr>
    <h3 style="text-align: center;">Features</h3>
    <ul style="font-size: 16px;">
        <li>Extraction and analysis of key resume attributes</li>
        <li>Graph-based analysis of recommendation networks</li>
        <li>Sentiment analysis of recommendation letters</li>
        <li>Fraud detection in recommendations</li>
        <li>Composite scoring system for candidate evaluation</li>
    </ul>
    """, 
    unsafe_allow_html=True
)