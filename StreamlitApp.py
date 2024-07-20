import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import json
import re


# Fetch the OpenAI API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Check if the OpenAI API key is set
if openai_api_key:
    print("OpenAI API key is set")
else:
    print("OpenAI API key is not set")


# Function to innitiate the LLM
def create_llm(llm_model):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    if openai_api_key:
        return ChatOpenAI(model=llm_model)
    else:
        print("OpenAI API key is not set")


# Function to innitate the embeddings
def create_embeddings(embedding_model):
    return OpenAIEmbeddings(model=embedding_model)


# Function to innitiate the text splitter
def create_text_splitter(doc):
    text_splitter = RecursiveCharacterTextSplitter()
    return text_splitter.split_text(doc)


# Function to innitiate the vector store
def create_vector_store(splitted_doc, embeddings):
    return FAISS.from_texts(splitted_doc, embeddings)


# Function to create the document chain
def build_document_chain(llm_model, prompt):
    return create_stuff_documents_chain(llm_model, prompt)


# Function to create a retrieval chain
def build_retrieval_chain(vector, document_chain):
    # Create the retriever
    retriever = vector.as_retriever()
    # Create the retrieval chain
    return create_retrieval_chain(retriever, document_chain)


# Function to return map name, map description, or map variables based on map id
def get_map_info(map_id, option, json_file = 'maps_intro.json'):
    with open(json_file) as file:
        data = json.load(file)
        for item in data:
            if (item["map_id"] == map_id and option == "name"):
                return item["map_name"]
            elif (item["map_id"] == map_id and option == "description"):
                return item["map_description"]
            elif (item["map_id"] == map_id and option == "variables"):
                return item["map_variables"]
    return None


# Function to create a prompt to classify whether an article is relevant to the diagram
def prompt_article_relevance(map_id):
    return f'''
**Background Information**\n
As a university professor researching child maltreatment, I would like your help analyzing a peer-reviewed article using a systematic or scoping review methodology. Your task involves thoroughly reviewing this document, and you will respond to my questions based on its content. Your insightful answers will be instrumental in developing causal loop diagrams. These diagrams are intended to systematically illustrate the various factors contributing to child maltreatment and the interconnections between these factors. Additionally, they will highlight mechanisms for preventing and addressing child maltreatment. The ultimate goal of these diagrams is to facilitate a deeper understanding of the complex dynamics involved in child maltreatment.\n                                        
**Task: Assessing Compatibility with Causal Loop Diagram**\n
Given the causal loop diagram titled "{get_map_info(map_id, option='name')}" and its detailed description: "{get_map_info(map_id, option='description')}", please assess whether the methodology and key findings of the review article align with this diagram.\n
For the article to be considered relevant to the diagram, it must investigate two or more of the variables listed below, including their interrelationships. Please note that the article may describe variables using different terminology than those listed. In such cases, you are expected to leverage your substantive domain knowledge and analytical skills to determine whether there is a match between the variables mentioned in the article and those listed below.\n
The variables and their specific definitions are:\n\n "{get_map_info(map_id, option='variables')}".\n\n
**Response Requirement:**\n
- Answer with a simple "Yes" or "No" to indicate whether the article aligns with the causal loop diagram based on the criteria provided.\n
- Briefly justify your decision in a seperate section.\n
- Comprehensive Review: Examine the entire article meticulously—from the title and abstract to the main text, including tables and figures—to gather information. Do not seek answers from the reference list.\n
- Autonomous Completion: Proceed to complete the tasks without requesting further confirmation. Your responses should be self-contained and decisive.\n
- Adherence to Format: Strictly follow the example response format provided, avoiding any additional comments, summaries, or deviations from the prescribed structure.\n
**Example Response Format**:\n
Decision: Yes/No\n
Justification: ...\n
'''


# Function to create a prompt to identify and summarize key variables
def prompt_key_relationships(map_id):
    return f'''
**Background Information**\n
As a university professor researching child maltreatment, I would like your help analyzing a peer-reviewed article using a systematic or scoping review methodology. Your task involves thoroughly reviewing this document, and you will respond to my questions based on its content. Your insightful answers will be instrumental in developing causal loop diagrams. These diagrams are intended to systematically illustrate the various factors contributing to child maltreatment and the interconnections between these factors. Additionally, they will highlight mechanisms for preventing and addressing child maltreatment. The ultimate goal of these diagrams is to facilitate a deeper understanding of the complex dynamics involved in child maltreatment.\n                                        
**Task: Identifying and Summarizing Key Variables in Relation to the Causal Loop Diagram**\n
Based on my assessment, the attached review article aligns with the causal loop diagram titled "{get_map_info(map_id, option='name')}". The description of the diagram follows: "{get_map_info(map_id, option='description')}" The diagram consists of the following variables:\n
"{get_map_info(map_id, option='variables')}".\n
Given this context, your tasks are now twofold:\n
1. Identify the Key Variables: Carefully pinpoint the specific variables discussed in the attached article from the above list provided. Please note that the article may describe variables using different terminology than that listed above. In such cases, you are expected to leverage your substantive domain knowledge and analytical skills to determine the closest match between the variables mentioned in the article and those provided in our list. This approach ensures accurate identification despite variations in terminology.\n
2. Summarize Their Relationships: For each pair of identified variables, succinctly describe how they are related. Ensure your summaries focus on dyadic (two-variable) relationships. If the article discusses correlations among three or more variables, please deconstruct these into dyadic relationships for the purpose of this analysis. In addition to providing a brief description of their relationship, specify whether the relationship is "directed" or "mutual". For "directed" relationships, clearly indicate the directionality by stating which variable impacts the other. If the relationship is "mutual", it indicates that both variables influence each other, and no directionality needs to be specified.\n
**Example Response Format**:\n
Key Variables Identified:\n
Variable 1: Description of Variable 1 as it relates to the article. "One or two quotes from the article."\n
Variable 2: Description of Variable 2 as it relates to the article. "One or two quotes from the article."\n
Variable 3: Description of Variable 3 as it relates to the article. "One or two quotes from the article."\n
...\n
Summarized Relationships Between Key Variables:\n
Variable 1 -> Variable 2 (Directed): Explanation of how Variable 1 directly influences Variable 2. "One or two quotes from the article."\n
Variable 2 <-> Variable 3 (Mutual): Explanation of how Variable 1 and Variable 3 mutually influence each other. "One or two quotes from the article."\n
...\n
**Response Requirement:**\n
- Focus on Existing Variables: Your identification of key variables and summaries of relationships between key variables should strictly adhere to the list of variables provided. Please avoid introducing any new variables not already mentioned.\n
- Dyadic Relationship Summaries: When summarizing relationships, maintain a strict focus on dyads, summarizing the connection between each pair of variables without extending into broader multi-variable correlations. Each summary should not only describe the connection between each pair of variables but also clarify whether the relationship is "directed" or "mutual". For directed relationships, specify the direction of influence.\n
- Use "->" to denote directed relationships and "<->" to denote mutual relationships.\n
- Comprehensive Review: Examine the entire article meticulously—from the title and abstract to the main text, including tables and figures—to gather information. Do not seek answers from the reference list.\n
- Autonomous Completion: Proceed to complete the tasks without requesting further confirmation. Your responses should be self-contained and decisive.\n
- Adherence to Format: Strictly follow the example response format provided, avoiding any additional comments, summaries, or deviations from the prescribed structure.\n
- Quoting Directly from the Article: Do not cite any external source. Instead, enrich your response by including one or two direct quotations from the article itself. These quotes should support the variables identified and the relationships summarized.\n
- Accuracy and Verification of Quotes: Carefully verify the accuracy and relevance of any quotes you include. Ensure that they are exact excerpts from the article and directly pertinent to the variables and relationships discussed. Avoid the fabrication of quotes or the inclusion of information not found in the article.\n
- Consistency between Key Variables and Their Relationships: Guarantee that every variable in the "Key Variable Identified" section is mentioned at least once in the "Summarized Relationships Between Key Variables" section, and conversely, that all relationships summarized in the "Summarized Relationships Between Key Variables" involve variables listed in the "Key Variables Identified" section.\n
'''


# Function to create a prompt to identify and summarize key variables
def prompt_additional_relationships(map_id):
    return f'''
**Background Information**\n
As a university professor researching child maltreatment, I would like your help analyzing a peer-reviewed article using a systematic or scoping review methodology. Your task involves thoroughly reviewing this document, and you will respond to my questions based on its content. Your insightful answers will be instrumental in developing causal loop diagrams. These diagrams are intended to systematically illustrate the various factors contributing to child maltreatment and the interconnections between these factors. Additionally, they will highlight mechanisms for preventing and addressing child maltreatment. The ultimate goal of these diagrams is to facilitate a deeper understanding of the complex dynamics involved in child maltreatment.\n                                        
**Task: Expanding on Variables and Relationships for Enhanced Diagram Context**\n
The attached review article aligns with the causal loop diagram titled "{get_map_info(map_id, option='name')}". The description of the diagram follows: "{get_map_info(map_id, option='description')}" The diagram consists of the following variables:\n
"{get_map_info(map_id, option='variables')}".\n
In the previous task, we had identified and summarized the key variables and their relationships from the provided list in the attached review article. Now we aim to delve deeper. Your next task involves uncovering and integrating additional variables not previously listed, alongside examining their interactions. This exploration is pivotal for revealing further complexity and interconnectivity, thereby enriching our causal loop diagram with nuanced insights and broadening its contextual framework to bolster its explanatory capability.\n
Task 1. Identification of Novel Variables: Scrutinize the review article to pinpoint variables that are distinct from those previously enumerated. This task transcends mere name matching; it requires the application of your domain expertise and analytical acumen to discern if the newly identified variables substantively diverge from the listed ones. Include these novel variables only if they represent a significant addition to our understanding.\n
Task 2. Summarize the Relationships Between Novel Variables: For each pair of newly identified variables, succinctly describe how they are related. Ensure your summaries focus on dyadic (two-variable) relationships. If the article discusses correlations among three or more variables, please deconstruct these into dyadic relationships for the purpose of this analysis. In addition to providing a brief description of their relationship, specify whether the relationship is "directed" or "mutual". For "directed" relationships, clearly indicate the directionality by stating which variable impacts the other. If the relationship is "mutual", it indicates that both variables influence each other, and no directionality needs to be specified.\n
Task 3: Summarize the Relationships Between Novel and Key Variables: Analogously, identify and summarize the relationships between novel variables and key variables included in the list.\n
**Example Response Format**:\n
Novel Variables Identified:\n
Variable A: Description of Variable A as it relates to the article. "One or two quotes from the article."\n
Variable B: Description of Variable B as it relates to the article. "One or two quotes from the article."\n
Variable C: Description of Variable C as it relates to the article. "One or two quotes from the article."\n
...\n
Summarized Relationships Between Novel Variables:\n
Variable A -> Variable B (Directed): Explanation of how Variable A directly influences Variable B. "One or two quotes from the article."\n
Variable B <-> Variable C (Mutual): Explanation of how Variable B and Variable C mutually influence each other. "One or two quotes from the article."\n
...\n
Summarized Relationships Between Novel Variables and Key Variables Included in the List:\n
Variable A -> Variable 1 (Mutual): Explanation of how Variable A directly influences Variable 1 (a variable included in the list). "One or two quotes from the article."\n
Variable B <-> Variable 2 (Mutual): Explanation of how Variable B and Variable 2 (a variable included in the list) mutually influence each other. "One or two quotes from the article."\n
...\n
**Response Requirement:**\n
- Dyadic Relationship Summaries: When summarizing relationships, maintain a strict focus on dyads, summarizing the connection between each pair of variables without extending into broader multi-variable correlations. Each summary should not only describe the connection between each pair of variables but also clarify whether the relationship is "directed" or "mutual". For directed relationships, specify the direction of influence.\n
- Use "->" to denote directed relationships and "<->" to denote mutual relationships.\n
- Comprehensive Review: Examine the entire article meticulously—from the title and abstract to the main text, including tables and figures—to gather information. Do not seek answers from the reference list.\n
- Autonomous Completion: Proceed to complete the tasks without requesting further confirmation. Your responses should be self-contained and decisive.\n
- Adherence to Format: Strictly follow the example response format provided, avoiding any additional comments, summaries, or deviations from the prescribed structure.\n
- Quoting Directly from the Article: Do not cite any external source. Instead, enrich your response by including one or two direct quotations from the article itself. These quotes should support the variables identified and the relationships summarized.\n
- Accuracy and Verification of Quotes: Carefully verify the accuracy and relevance of any quotes you include. Ensure that they are exact excerpts from the article and directly pertinent to the variables and relationships discussed. Avoid the fabrication of quotes or the inclusion of information not found in the article.\n
'''


# Function to remove referneces from the text (type: str) of a file.
def remove_references(doc_text):

    # Use the regex pattern for identifying the reference section
    pattern = re.compile(r'(?i)\n\s*(References|Bibliography|)\s*\n', re.DOTALL)

    # Search for the reference section
    match = pattern.search(doc_text)
    reference_section_text = None
    if match:
        # Extract the reference section starting from the match position
        reference_section_start = match.start()
        reference_section_text = doc_text[reference_section_start:]
        print("Reference section found")
        # If a match is found, trim the document text up to the start of "References"
        trimmed_doc_text = doc_text[:match.start()]
    else:
        print("Reference section not found.")
        # If no "References" section is found, keep the entire document text as is
        trimmed_doc_text = doc_text

    # Return the trimmed document text
    return trimmed_doc_text, reference_section_text


# Overall function to create the pipeline from making a query to obtaining answers
def create_query_pipeline(llm_model, doc_text, embedding_model, map_id, query_option):
    # Initialize the LLM
    llm = create_llm(llm_model=llm_model)

    # Remove references, if any
    doc_text, removed_references = remove_references(doc_text=doc_text)

    # Split the document
    splitted_doc = create_text_splitter(doc=doc_text)

    # Innitiate the embeddings
    embeddings = create_embeddings(embedding_model=embedding_model)

    # Create the vector store
    vector = create_vector_store(splitted_doc=splitted_doc, embeddings=embeddings)

    # Create the prompt
    # Important note: This prompt is provided by the Langchain and should not be modified. Otherwise, we will run
    # into issues that the retrieval chain will not be able to extract the relevant information from the document.
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context: {context}
    Question: {input}
    """)

    # Create the document chain
    document_chain = build_document_chain(llm_model=llm, prompt=prompt)

    # Create the retrieval chain
    retrieval_chain = build_retrieval_chain(vector=vector, document_chain=document_chain)

    # Create the query
    if query_option == "article_relevance":
        query = prompt_article_relevance(map_id=map_id)
    elif query_option == "key_relationships":
        query = prompt_key_relationships(map_id=map_id)
    elif query_option == "additional_relationships":
        query = prompt_additional_relationships(map_id=map_id)
    else:
        ValueError("Invalid query option. Please choose 'article_relevance', 'key_relationships', or 'additional_relationships'.")
    
    # Invoke the retrieval chain
    response = retrieval_chain.invoke({"input": query})
    return response['answer'], removed_references


def main():
    # Set page configuration
    st.set_page_config(page_title="CICM Mapping - Article Evaluation", page_icon="")
    st.header("CICM Mapping - Article Evaluation")

    # Upload file
    pdf_file = st.file_uploader("Upload an artcile (PDF file)", type="pdf")

    # Model selection dropdown
    model_options = ["gpt-4-turbo", "gpt-4o", "gpt-4", "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-1106-preview"]
    selected_model = st.selectbox("Choose a GPT-4 model", model_options)

    # Embedding model selection dropdown
    embedding_model_options = ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
    selected_embedding_model = st.selectbox("Choose an OpenAI embedding model", embedding_model_options)

    # Map ID selection dropdown
    map_id_options = [2, 4]
    selected_map_id = st.selectbox("Choose a map ID", map_id_options)

    # Query option selection dropdown
    query_options = ["article_relevance", "key_relationships", "additional_relationships"]
    selected_query_option = st.selectbox("Choose query option", query_options)

    # Submit button
    submit_button = st.button("Submit")

    # Extract the text from the pdf file
    doc_text = ""
    if submit_button and pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            doc_text += page.extract_text()

        # Build the pipeline 
        answer_relevance, remove_references = create_query_pipeline(
            llm_model=selected_model,
            doc_text=doc_text,
            embedding_model=selected_embedding_model,
            map_id=selected_map_id,
            query_option=selected_query_option)

        # Display the answer
        st.write(answer_relevance)

if __name__ == '__main__':
    main()
