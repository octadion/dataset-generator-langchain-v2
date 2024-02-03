import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
load_dotenv()
prompt_template_questions = """
You are an expert in the field of law capable of creating questions based on legal text.
Your goal is to create a legal dataset. You do this by creating questions about the text below:

------------
{text}
------------

Create questions that will assist in the creation of a legal dataset. Make sure no important information is missed. Please generate the questions in Indonesian language.

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = """
You are an expert in the field of law capable of creating questions based on legal text.
Your goal is to create a legal dataset.
We have received some legal questions to a certain extent: {existing_answer}.
We have the option to refine these existing legal questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original legal questions in Indonesian.
If the context is not helpful, please provide the original legal questions.

Please note: The refined legal questions should be generated in Indonesian language.

QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)

st.title('DataGen-v2')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

file_path = None

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

if file_path:
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_question_gen = ''
    for page in data:
        text_question_gen += page.page_content

    text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

    llm_question_gen = ChatOpenAI()

    question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                              question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

    questions = question_gen_chain.run(docs_question_gen)

    llm_answer_gen = ChatOpenAI()

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(docs_question_gen, embeddings)

    answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff",
                                                   retriever=vector_store.as_retriever(k=2))

    question_list = questions.split("\n")

    question_answer_pairs = []

    for question in question_list:
        st.write("Question: ", question)
        answer = answer_gen_chain.run(question)
        question_answer_pairs.append([question, answer])
        st.write("Answer: ", answer)
        st.write("--------------------------------------------------\n\n")

    answers_dir = os.path.join(tempfile.gettempdir(), "answers")
    os.makedirs(answers_dir, exist_ok=True)

    qa_df = pd.DataFrame(question_answer_pairs, columns=["Question", "Answer"])

    csv_file_path = os.path.join(answers_dir, "questions_and_answers.csv")

    if os.path.isfile(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        merged_df = pd.concat([existing_df, qa_df], ignore_index=True)
    else:
        merged_df = qa_df
    merged_df.to_csv(csv_file_path, index=False)

    st.markdown('### Download Questions and Answers in CSV')
    st.download_button("Download Questions and Answers (CSV)", csv_file_path)

if file_path:
    os.remove(file_path)