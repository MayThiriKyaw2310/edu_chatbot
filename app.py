import streamlit as st
from main import initialize_gpt, query_with_language

api_key = st.secrets.get("OPENAI_API_KEY")

def main():
    st.title("Education Chatbot")
    st.write("Ask questions about education and career. The chatbot will automatically detect the language.")

    if not api_key:
        st.error("API key is missing! Please configure it in the Streamlit Secrets Manager.")
        return

    # Custom CSS for UI
    st.markdown("""
        <style>
            .user {
                text-align: right;
                margin-bottom: 15px;
                background-color: #e1f5fe;
                padding: 10px 20px;
                border-radius: 25px;
                max-width: 75%;
                margin-left: auto;
            }

            .bot {
                text-align: left;
                margin-bottom: 15px;
                background-color: #f1f1f1;
                padding: 10px 20px;
                border-radius: 25px;
                max-width: 75%;
            }
        </style>
    """, unsafe_allow_html=True)

    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    def submit_question():
        question = st.session_state["question"]
        if question:
            llm = initialize_gpt(api_key)
            if not llm:
                st.error("Failed to initialize the GPT model.")
                return
            
            context = "Education-related information"
            try:
                response = query_with_language(
                    llm,
                    question,
                    context,
                    "\n".join(
                        [f"{chat['role']}: {chat['message']}" for chat in st.session_state["conversation_history"]]
                    )
                )

                st.session_state["conversation_history"].append({"role": "user", "message": question})
                st.session_state["conversation_history"].append({"role": "bot", "message": response})
            except Exception as e:
                st.error(f"Error querying the GPT model: {e}")
            st.session_state["question"] = ""

    if "conversation_history" in st.session_state:
        for chat in st.session_state["conversation_history"]:
            if chat["role"] == "user":
                st.markdown(f'<div class="user">{chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot">{chat["message"]}</div>', unsafe_allow_html=True)

    st.text_input(
        "Enter your question here:",
        key="question",
        on_change=submit_question,
        label_visibility="collapsed"
    )

if __name__ == "__main__":
    main()
