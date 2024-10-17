from django.shortcuts import render

# Create your views here.
import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile
import requests
import json
from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory



# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"



def main():
    st.set_page_config(
        page_title="RAG Chat")

    st.title("mySUNI RAG Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"),
            key="model_selection"
        )
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = os.getenv('openai_api_key')
        
        # 환경 변수 입력을 위한 UI 추가
        langchain_api_key =os.getenv('langchain_api_key')
        langchain_project = os.getenv('langchain_project')
        
        process = st.button("Process")
    
    # 입력받은 환경변수로 설정
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project


    if process:
        if not openai_api_key or not langchain_api_key or not langchain_project:
            st.info("Please add all necessary API keys and project information to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! mysuni RAG chatbot 입니다. 주어진 문서에 대해 궁금한 점을 물어보세요."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("Message to chatbot"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def check_cmi_subs(svc_tel_no):
    url =  os.getenv('url')
    headers = {
        "Authorization": os.getenv('token'),
        "Content-Type": "application/json",
        "x-apigw-api-id": os.getenv('api_id'),
    }
    body = dict()
    body["svc_tel_no"] = svc_tel_no #01023539109
    print(body)
    response = requests.post(url, data=json.dumps(body), headers=headers)

    result = json.loads(response.content)
    print(result)
    return result

ai_key = os.getenv('openai_api_key')
client = OpenAI(
    api_key = ai_key,
)

# 음성을 텍스트로 변환하는 함수
def voice_to_text(files):

    print("voice_to_text")
    client = OpenAI(api_key=ai_key)
    audio_file =files
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file
    )
    transcription = transcript.text
    print(transcription)

    ## 텍스트 교정
    system_prompt = '''
        You are a helpful assistant for the company SK 7mobile.
        and You are highly skilled AI trained in language comprehension and summarization.
        I would like to read the follwing text and summarize it into a concis abstract paragraph.
        The conversation is between a customer service agent and the customer.
        Please summarize the conversation so that what the customer service agent and the customer said can be distinguished from each other.
        and Please write the customer's request as the title and Summarize the topics of the conversation in bullet points as summary.
        If there is a phone number in the conversation, change it to the following format: {"svc_tel_no" : "010xxxxxxxx"}
        Replace the phone number with the actual number found in the conversation.
        all answer in Korean
        Aim to retain the most important points,
        providing a coherent and readable summary that colud help a person understand the main points of the discussion without needing to read the entire text.
        please avoid unnecessary details or tangential points.
        '''

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def get_my_data(v_text):
        print("call_my_Data")
        ### 1. 음성을 텍스트로 변환
        # v_text = voice_to_text()

        ### 2. GPT 에 호출할 데이터를 정의한다.
        messages = [
            {
                "role": "user",
                "content": v_text
            },
            {
                "role": "system",
                "content": "CMI Bot"
            }
        ]
        functions = [
            {
                "name": "check_cmi_subs",
                "description": "Check CMI Subscriber",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "svc_tel_no": {
                            "type": "string",
                            "description": "Service TelNumber",
                        }
                    },
                    "required": ["svc_tel_no"],
                },
            }
        ]

        ### 3. OpenAI API에 대화와 함수 정보를 전달하고 응답을 확인한다.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        ### 4. 응답 확인 후
        response_message = response.choices[0].message
        print("첫번재 응답")
        print(response)

        ### 5. GPT 모델의 응답에서 함수 호출 여부를 확인
        if response_message.function_call:
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "check_cmi_subs": check_cmi_subs,
            }
            function_name = response_message.function_call.name
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message.function_call.arguments)
            function_response = fuction_to_call(
                svc_tel_no=function_args['svc_tel_no'],
            )

            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response['rps_msg'],
                }
            )
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )  # get a new response from GPT where it can see the function response

        return second_response.choices[0].message.content

def tiktoken_len(text):
    """
    주어진 텍스트에 대한 토큰 길이를 계산합니다.

    Parameters:
    - text: str, 토큰 길이를 계산할 텍스트입니다.

    Returns:
    - int, 계산된 토큰 길이입니다.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_document(doc):
    """
    업로드된 문서 파일을 로드하고, 해당 포맷에 맞는 로더를 사용하여 문서를 분할합니다.

    지원되는 파일 유형에 따라 적절한 문서 로더(PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader)를 사용하여
    문서 내용을 로드하고 분할합니다. 지원되지 않는 파일 유형은 빈 리스트를 반환합니다.

    Parameters:
    - doc (UploadedFile): Streamlit을 통해 업로드된 파일 객체입니다.

    Returns:
    - List[Document]: 로드 및 분할된 문서 객체의 리스트입니다. 지원되지 않는 파일 유형의 경우 빈 리스트를 반환합니다.
    """
    # 임시 디렉토리에 파일 저장
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    # 파일 쓰기
    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())  # 파일 내용을 임시 파일에 쓴다

    # 파일 유형에 따라 적절한 로더를 사용하여 문서 로드 및 분할
    try:
        if file_path.endswith('.pdf'):
            loaded_docs = PyPDFLoader(file_path).load_and_split()
        elif file_path.endswith('.docx'):
            loaded_docs = Docx2txtLoader(file_path).load_and_split()
        elif file_path.endswith('.pptx'):
            loaded_docs = UnstructuredPowerPointLoader(file_path).load_and_split()
        else:
            loaded_docs = []  # 지원되지 않는 파일 유형
    finally:
        os.remove(file_path)  # 작업 완료 후 임시 파일 삭제

    return loaded_docs

def get_text(docs):
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list


def get_text_chunks(text):
    """
    주어진 텍스트 목록을 특정 크기의 청크로 분할합니다.

    이 함수는 'RecursiveCharacterTextSplitter'를 사용하여 텍스트를 청크로 분할합니다. 각 청크의 크기는
    `chunk_size`에 의해 결정되며, 청크 간의 겹침은 `chunk_overlap`으로 조절됩니다. `length_function`은
    청크의 실제 길이를 계산하는 데 사용되는 함수입니다. 이 경우, `tiktoken_len` 함수가 사용되어 각 청크의
    토큰 길이를 계산합니다.

    Parameters:
    - text (List[str]): 분할할 텍스트 목록입니다.

    Returns:
    - List[str]: 분할된 텍스트 청크의 리스트입니다.

    사용 예시:
    텍스트 목록이 주어졌을 때, 이 함수를 호출하여 각 텍스트를 지정된 크기의 청크로 분할할 수 있습니다.
    이렇게 분할된 청크들은 텍스트 분석, 임베딩 생성, 또는 기계 학습 모델의 입력으로 사용될 수 있습니다.


    주의:
    `chunk_size`와 `chunk_overlap`은 분할의 세밀함과 처리할 텍스트의 양에 따라 조절할 수 있습니다.
    너무 작은 `chunk_size`는 처리할 청크의 수를 불필요하게 증가시킬 수 있고, 너무 큰 `chunk_size`는
    메모리 문제를 일으킬 수 있습니다. 적절한 값을 실험을 통해 결정하는 것이 좋습니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    주어진 텍스트 청크 리스트로부터 벡터 저장소를 생성합니다.

    이 함수는 Hugging Face의 'jhgan/ko-sroberta-multitask' 모델을 사용하여 각 텍스트 청크의 임베딩을 계산하고,
    이 임베딩들을 FAISS 인덱스에 저장하여 벡터 검색을 위한 저장소를 생성합니다. 이 저장소는 텍스트 청크들 간의
    유사도 검색 등에 사용될 수 있습니다.

    Parameters:
    - text_chunks (List[str]): 임베딩을 생성할 텍스트 청크의 리스트입니다.

    Returns:
    - vectordb (FAISS): 생성된 임베딩들을 저장하고 있는 FAISS 벡터 저장소입니다.

    모델 설명:
    'jhgan/ko-sroberta-multitask'는 문장과 문단을 768차원의 밀집 벡터 공간으로 매핑하는 sentence-transformers 모델입니다.
    클러스터링이나 의미 검색 같은 작업에 사용될 수 있습니다. KorSTS, KorNLI 학습 데이터셋으로 멀티 태스크 학습을 진행한 후,
    KorSTS 평가 데이터셋으로 평가한 결과, Cosine Pearson 점수는 84.77, Cosine Spearman 점수는 85.60 등을 기록했습니다.
"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key, model_selection):
    """
    대화형 검색 체인을 초기화하고 반환합니다.

    이 함수는 주어진 벡터 저장소, OpenAI API 키, 모델 선택을 기반으로 대화형 검색 체인을 생성합니다.
    이 체인은 사용자의 질문에 대한 답변을 생성하는 데 필요한 여러 컴포넌트를 통합합니다.

    Parameters:
    - vetorestore: 검색을 수행할 벡터 저장소입니다. 이는 문서 또는 데이터를 검색하는 데 사용됩니다.
    - openai_api_key (str): OpenAI API를 사용하기 위한 API 키입니다.
    - model_selection (str): 대화 생성에 사용될 언어 모델을 선택합니다. 예: 'gpt-3.5-turbo', 'gpt-4-turbo-preview'.

    Returns:
    - ConversationalRetrievalChain: 초기화된 대화형 검색 체인입니다.

    함수는 다음과 같은 작업을 수행합니다:
    1. ChatOpenAI 클래스를 사용하여 선택된 모델에 대한 언어 모델(LLM) 인스턴스를 생성합니다.
    2. ConversationalRetrievalChain.from_llm 메소드를 사용하여 대화형 검색 체인을 구성합니다. 이 과정에서,
       - 검색(retrieval) 단계에서 사용될 벡터 저장소와 검색 방식
       - 대화 이력을 관리할 메모리 컴포넌트
       - 대화 이력에서 새로운 질문을 생성하는 방법
       - 검색된 문서를 반환할지 여부 등을 설정합니다.
    3. 생성된 대화형 검색 체인을 반환합니다.
    """
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)

    prompt = '''
        You are a helpful assistant for the company SK 7mobile.
        and You are highly skilled AI trained in language comprehension and summarization.
        I would like to read the follwing text and summarize it into a concis abstract paragraph.
        The conversation is between a customer service agent and the customer.
        Please summarize the conversation so that what the customer service agent and the customer said can be distinguished from each other.
        and Please write the customer's request as the title and Summarize the topics of the conversation in bullet points as summary.
        If there is a phone number in the conversation, change it to the following format: {"svc_tel_no" : "010xxxxxxxx"}
        Replace the phone number with the actual number found in the conversation.
        all answer in Korean
        Aim to retain the most important points,
        providing a coherent and readable summary that colud help a person understand the main points of the discussion without needing to read the entire text.
        please avoid unnecessary details or tangential points.

    '''
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return conversation_chain


# def get_conversation_chain(v_text, openai_api_key, model_selection):
#     """
#     대화형 검색 체인을 초기화하고 반환합니다.

#     이 함수는 주어진 벡터 저장소, OpenAI API 키, 모델 선택을 기반으로 대화형 검색 체인을 생성합니다.
#     이 체인은 사용자의 질문에 대한 답변을 생성하는 데 필요한 여러 컴포넌트를 통합합니다.

#     Parameters:
#     - vetorestore: 검색을 수행할 벡터 저장소입니다. 이는 문서 또는 데이터를 검색하는 데 사용됩니다.
#     - openai_api_key (str): OpenAI API를 사용하기 위한 API 키입니다.
#     - model_selection (str): 대화 생성에 사용될 언어 모델을 선택합니다. 예: 'gpt-3.5-turbo', 'gpt-4-turbo-preview'.

#     Returns:
#     - ConversationalRetrievalChain: 초기화된 대화형 검색 체인입니다.

#     함수는 다음과 같은 작업을 수행합니다:
#     1. ChatOpenAI 클래스를 사용하여 선택된 모델에 대한 언어 모델(LLM) 인스턴스를 생성합니다.
#     2. ConversationalRetrievalChain.from_llm 메소드를 사용하여 대화형 검색 체인을 구성합니다. 이 과정에서,
#        - 검색(retrieval) 단계에서 사용될 벡터 저장소와 검색 방식
#        - 대화 이력을 관리할 메모리 컴포넌트
#        - 대화 이력에서 새로운 질문을 생성하는 방법
#        - 검색된 문서를 반환할지 여부 등을 설정합니다.
#     3. 생성된 대화형 검색 체인을 반환합니다.
#     """
#     llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
#   ### 2. GPT 에 호출할 데이터를 정의한다.
#     messages = [
#         {
#             "role": "user",
#             "content": v_text
#         },
#         {
#             "role": "system",
#             "content": "CMI Bot"
#         }
#     ]
#     functions = [
#         {
#             "name": "check_cmi_subs",
#             "description": "Check CMI Subscriber",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "svc_tel_no": {
#                         "type": "string",
#                         "description": "Service TelNumber",
#                     }
#                 },
#                 "required": ["svc_tel_no"],
#             },
#         }
#     ]

#     ### 3. OpenAI API에 대화와 함수 정보를 전달하고 응답을 확인한다.
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         functions=functions,
#         function_call="auto",
#     )

#     ### 4. 응답 확인 후
#     response_message = response.choices[0].message
#     print("첫번재 응답")
#     print(response)

#     ### 5. GPT 모델의 응답에서 함수 호출 여부를 확인
#     if response_message.function_call:
#         # Note: the JSON response may not always be valid; be sure to handle errors
#         available_functions = {
#             "check_cmi_subs": check_cmi_subs,
#         }
#         function_name = response_message.function_call.name
#         fuction_to_call = available_functions[function_name]
#         function_args = json.loads(response_message.function_call.arguments)
#         function_response = fuction_to_call(
#             svc_tel_no=function_args['svc_tel_no'],
#         )

#         messages.append(response_message)
#         messages.append(
#             {
#                 "role": "function",
#                 "name": function_name,
#                 "content": function_response['rps_msg'],
#             }
#         )
#         second_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#         )  # get a new response from GPT where it can see the function response
#     return second_response.choices[0].message.content



if __name__ == '__main__':
    main()