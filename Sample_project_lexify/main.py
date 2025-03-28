import streamlit as st
import pdfplumber
from GenAIsis.Builder.ChatLLM import ChatLLMBuilder

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

st.title("ZIF Lexify")
st.subheader("Your Swift and Insightful Legal briefing Assistant")

uploaded_file = st.file_uploader("Upload a Contract In PDF Format", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from the contract..."):
        contract_text = extract_text_from_pdf(uploaded_file)
    
    
    if st.button("Analyze Contract"):
        with st.spinner("Analyzing contract using google vertexai..."):
            formatted_prompt = f"""
            Analyze the attached contract PDF for potential risks, missing clauses, and non-standard clauses.  Specifically:

                1. **Identify potential risks:**  For each clause, highlight any potential legal or business risks. Explain the nature of the risk and suggest mitigation strategies. Consider risks related to:
                    * Ambiguity in language that could lead to disputes
                    * Unfavorable terms the client (e.g., limitation of liability, indemnification)
                    * Payment terms
                    * Termination clauses
                    * Intellectual property rights
                    * Dispute resolution
                    * Governing law
                    * Confidentiality
                    * Force majeure

                2. **Missing clauses:** Identify any standard or essential clauses that are missing from the contract. Explain why these clauses are important and what risks their absence creates.  Consider common missing clauses such as:
                    * Entire agreement clause
                    * Severability clause
                    * Assignment clause
                    * No waiver clause
                    * Notice provisions

                3. **Non-standard clauses:** Identify any clauses that deviate significantly from standard practice for contracts of this type (please specify the type of contract if known, e.g., "software license agreement," "real estate lease"). Explain why these clauses are unusual and what potential risks they pose.

                4. **Risk assessment of specific clauses:** For each clause identified as posing a risk (in point 1 or 3), provide a detailed risk assessment.  This should include:
                    * The specific wording of the clause
                    * The nature of the risk
                    * The potential impact of the risk
                    * Suggested mitigation strategies

                5. **Overall risk assessment:** Provide a summary of the overall risk level of the contract (e.g., low, medium, high) based on the identified risks and missing clauses.

                Provide your analysis in a  clear and organized format, ideally using bullet points or a table .  Cite the specific clause numbers or sections when discussing them, and seperate them as High,Medium and low risk based on the above points.
                If the content is not related to the contract,Dont Generate the analysis.
                            
            **Contract Content:**
            {contract_text}
            
            """
            
            chat_llm_builder = ChatLLMBuilder()
            chat_llm = chat_llm_builder.set_prompt(
                template=formatted_prompt
            ).llm_chat.chat_google_vertexai(max_tokens=8190).build()
            
            response = chat_llm
            
            st.subheader("Analysis Result")
            st.write(response)