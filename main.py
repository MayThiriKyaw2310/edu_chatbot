import re
import unicodedata
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

def normalize_text(text):
    return unicodedata.normalize("NFC", text.strip())

api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("API key not found. Please add it to the secrets.toml file or Streamlit Cloud Secrets Management.")
    st.stop()

# Initialize GPT model
def initialize_gpt(api_key, model_name="gpt-4"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.2,
        openai_api_key=api_key
    )

# Define prompts for Burmese and English
burmese_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        သင်သည် ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ မေးခွန်းများအတွက် ကျောင်းသားများကို အကြံဉာဏ်ပေးသူ ဖြစ်ပါသည်။ အတိုချုံးပြီး အရေးပါသည့် အချက်များကိုသာ ပေးပါ။ conversation history နှင့် context အပေါ် အခြေခံပြီး တိုတို ရိုးရှင်းသော ဖြေကြားချက်ပေးပါ။

        စကားဝိုင်းမှတ်တမ်း:
        {conversation_history}

        အခြေခံချက်: {context}

        မေးခွန်း: {question}

        ဖြေကြားချက်: (တိုတောင်းပြီး အရေးပါသော အချက်များပဲ ရေးပါ။ များစွာ မရေးပါနဲ့)
    """
)

english_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        You are an educational advisor helping students with their academic and career-related questions. Provide concise and relevant answers based on the conversation history and context.

        Conversation History:
        {conversation_history}

        Context: {context}

        Question: {question}

        Answer:
    """
)

# Query function with language detection
def query_with_language(llm, question, context="", conversation_history=""):
    try:
        question = normalize_text(question)
        conversation_history = normalize_text(conversation_history)
        context = normalize_text(context)

        # Detect language using Unicode range for Burmese
        if re.search(r"[\u1000-\u109F]", question):
            language = "burmese"
        else:
            language = "english"

        # Select appropriate prompt
        if language == "burmese":
            formatted_prompt = burmese_prompt.format(
                conversation_history=conversation_history, question=question, context=context
            )
        else:
            formatted_prompt = english_prompt.format(
                conversation_history=conversation_history, question=question, context=context
            )

        # Invoke GPT model
        response = llm.invoke(formatted_prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    except Exception as e:
        return f"An error occurred: {str(e)}"

st.title("Education Chatbot")
st.write("Ask questions about education and career. The chatbot will automatically detect the language.")

question = st.text_input("Enter your question:")
context = st.text_area("Provide additional context (optional):")
conversation_history = st.text_area("Conversation history (optional):")

llm = initialize_gpt(api_key)

if question:
    st.write("### Debug Info:")
    st.write(f"Normalized Question: {question}")
    st.write(f"Detected Language: {'Burmese' if re.search(r'[\u1000-\u109F]', question) else 'English'}")

    response = query_with_language(llm, question, context, conversation_history)
    st.write("### Response:")
    st.write(response)
