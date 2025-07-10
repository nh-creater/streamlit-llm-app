from dotenv import load_dotenv
load_dotenv()

# app.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# --- LLMからの回答を返す関数を定義 ---
def invoke_llm(user_input: str, expert_type: str) -> str:
    """
    ユーザーの入力と選択された専門家タイプに基づいてLLMを呼び出し、回答を返します。

    Args:
        user_input (str): ユーザーからの入力テキスト。
        expert_type (str): 選択された専門家の種類（例: "技術専門家", "ビジネス戦略家"）。

    Returns:
        str: LLMからの回答テキスト。
    """
    # 専門家タイプに応じたシステムメッセージを定義
    system_message = ""
    if expert_type == "技術専門家":
        system_message = "あなたは最先端の技術動向に精通した技術専門家です。質問に対して、技術的な観点から詳細かつ正確な情報を提供してください。"
    elif expert_type == "ビジネス戦略家":
        system_message = "あなたは市場分析と事業戦略の策定に長けたビジネス戦略家です。質問に対して、ビジネス的な視点から実用的で洞察に富んだアドバイスを提供してください。"
    else:
        # 未定義の専門家タイプが選択された場合のデフォルトまたはエラーハンドリング
        system_message = "あなたは一般的なアシスタントです。質問に対して、丁寧かつ分かりやすく回答してください。"

    # プロンプトテンプレートを作成
    # system_messageには選択された専門家タイプに応じた指示が含まれます
    # human_messageにはユーザーからの実際の質問が含まれます
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input_text}"),
        ]
    )

    # LLMモデルを初期化
    # 環境変数 OPENAI_API_KEY が設定されている必要があります
    # Streamlit Community Cloudにデプロイする際は、Settings -> Secrets で設定してください
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # チェーンを作成し、LLMを呼び出す
    # promptにユーザーの入力テキストを渡します
    chain = prompt | llm
    response = chain.invoke({"input_text": user_input})

    # LLMの回答内容を返す
    return response.content

# --- WebアプリのUI設定 ---
st.set_page_config(page_title="LLM専門家チャットアプリ", layout="centered")

st.title("🤖 LLM専門家チャットアプリ")

# アプリの概要と操作方法
st.markdown(
    """
    このアプリは、LangChainとLLM（大規模言語モデル）を利用して、
    選択した専門家の視点から質問に回答します。
    
    ### 使い方
    1.  **専門家を選択**: 下のラジオボタンから、LLMに振る舞わせたい専門家の種類を選びます。
    2.  **質問を入力**: テキストボックスに質問や相談内容を入力します。
    3.  **回答を取得**: 「回答を生成」ボタンをクリックすると、選択した専門家が回答を生成します。
    """
)

st.divider()

# ラジオボタンで専門家の種類を選択
st.subheader("1. 専門家を選択してください")
expert_selection = st.radio(
    "LLMに振る舞わせる専門家を選んでください:",
    ("技術専門家", "ビジネス戦略家"),
    index=0, # デフォルトで「技術専門家」を選択
    key="expert_radio"
)

st.subheader("2. 質問を入力してください")
# 入力フォーム
user_query = st.text_area(
    "ここに質問や相談内容を入力してください:",
    placeholder="例: 量子コンピュータの最新の進展について教えてください。",
    height=150,
    key="user_query_input"
)

# 回答生成ボタン
if st.button("回答を生成", key="generate_button"):
    if user_query:
        with st.spinner("専門家が回答を生成中です..."):
            try:
                # 定義した関数を呼び出し、LLMから回答を取得
                llm_response = invoke_llm(user_query, expert_selection)
                st.subheader("3. 専門家からの回答")
                st.info(llm_response) # 回答を情報ボックスで表示
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.warning("APIキーが正しく設定されているか確認してください。")
    else:
        st.warning("質問内容を入力してください。")

st.divider()
st.caption("Powered by LangChain & Streamlit")