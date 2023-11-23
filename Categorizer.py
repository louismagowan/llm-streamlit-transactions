# Normal imports
import pandas as pd
import streamlit as st
import os


# Langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="ChatGPT App - Categorize Transactions",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title('Categorize Your Transactions')

# create bool to indicate success for later on
api_token_success = False

# -------------------------- SIDEBAR ------------------------------------------
# API Credentials
with st.sidebar:
    st.title('ChatGPT Categorizer')
    st.write('This chatbot is created using the [gpt-4-1106-preview](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) model from ChatGPT')
    if 'API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        api_token = st.secrets['API_TOKEN']
        # create bool to indicate success for later on
        api_token_success = True
    else:
        api_token = st.text_input('Enter API token:', type='password')
        if not len(api_token)==51:
            st.warning('Please enter the correct credentials!', icon='⚠️')
            api_token_success = False
        else:
            st.success('Please proceed to categorizing your transactions', icon='👉')
            # create bool to indicate success for later on
            api_token_success = True

    st.markdown("Please select how creative you want the bot to be with its guesses.  \
                \n The higher the temperature, the more creative the bot will be. The lower the temperature, the more consistent the bot will be. \
                Suggested value is 0.1")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    os.environ['API_TOKEN'] = api_token

# -------------------------- PROMPT AND DATA UPLOAD -----------------------------------------

st.markdown("# Load and Select Your Data")
st.markdown("**:blue[Please upload your transaction data below and then inspect it]**")

# Add the data you want to be classified
uploaded_file = st.file_uploader("**Upload transactions data**")
# Display the data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Drop the label column, since this is wrong
    df = df.drop(columns=['Label'])
    # Highlight the spend column green and make that writing white
    st.dataframe(df.style.set_properties(subset=['AVG_SPEND_EUR'], **{'color': 'green'}))

    # Get user to select the row they want to categorise
    row_number = st.number_input("**:red[Enter the row number of the transaction you want to categorise]**", min_value=0, max_value=len(df)-1, value=0, step=1)
    # Display the row
    row = df.iloc[row_number]
    display_format_row = pd.DataFrame(row).T.style.set_properties(subset=['AVG_SPEND_EUR'], **{'color': 'green'})
    st.write("The row you selected is: ", display_format_row)
st.markdown("#")


# Give the default list of categories to sort transactions into
default_categories = [
    "marketing",
    "legal_and_accounting",
    "tax",
    "transport",
    "office_rental",
    "salary",
    "fees",
    "food_and_grocery",
    "it_and_electronics",
    "insurance",
    "finance",
    "manufacturing",
    "other_expense",
    "hardware_and_equipment",
    "utility",
    "sales",
    "treasury_and_interco",
    "logistics",
    "other_income",
    "hotel_and_lodging",
    "other_service",
    "restaurant_and_bar",
    "office_supply",
    "atm",
    "subscription",
    "gas_station",
    "online_service"
]

## Create prompts
# Create system prompt
system_template = (
    "You are a helpful assistant that categorises bank transaction data into meaningful groups, allowing people to see what categories they are spending on."
    f"The categories are the following: {default_categories}"
    "Give your answer in the following EXACT format: 'Category Prediction: <category> \n 'Backup Prediction: <category>'"
    "Where Category Prediction is your best guess and Backup Prediction is the category you think is 2nd most likely and taking a newline on the '\n'"
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Create user prompt
human_template = (
    "My transaction data is as follows: "
    "COUNTERPARTY_NAME is {counterparty_name} "
    "MCC_CODE is {mcc_code} "
    "OPERATION_TYPE is {operation_type} "
    "AVG_SPEND_EUR is {avg_spend_eur} "
                 )

st.markdown("# Create Custom Categories")
# Add the user specified categories
st.markdown("**:blue[Please any custom categories you want to add]**")
user_specified_category_1 = st.text_input("Enter your 1st custom category")
user_specified_category_2 = st.text_input("Enter your 2nd custom category")


if user_specified_category_1:
        human_template = human_template + (
            "I also would like you to add these additional categories to sort the transaction into, on top of the ones mentioned earlier:"
            f"{[user_specified_category_1, user_specified_category_2]}"
        )
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# Create memory prompt
memory_template = (
    f"Also, categorise all transactions where COUNTERPARTY_NAME is like 'Adidas' as {user_specified_category_1}"
)
memory_message_prompt = SystemMessagePromptTemplate.from_template(memory_template)

# Generate prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt,
     memory_message_prompt,
     human_message_prompt
    ]
)

# Format prompt using the row data
formatted_prompt = chat_prompt.format_prompt(
    counterparty_name=row["COUNTERPARTY_NAME"],
    mcc_code=row["MCC_CODE"],
    operation_type=row["OPERATION_TYPE"],
    avg_spend_eur=row["AVG_SPEND_EUR"]
).to_messages()

# Display the unformatted prompt if user wants it
st.markdown("**Would you like to show your raw prompt which will be sent to ChatGPT?** \
             \n  _It's a bit messy-looking, but it'll run just fine!_")
display_raw_prompt = st.checkbox('Yes please! :pray:')
if display_raw_prompt:
    st.write("Your raw prompt looks like:   \n ", chat_prompt)

# Display the formatted prompt if user wants it
st.markdown("**Would you like to show your formatted prompt which will be sent to ChatGPT?** \
             \n  _It's even messier, but it'll also run just fine!_ 😉")
display_formatted_prompt = st.checkbox('Yes please! :pray:', key="formatted_prompt")
if display_formatted_prompt:
    st.write("Your formatted prompt looks like:   \n ", formatted_prompt)

# -------------------------- MODEL -----------------------------------------

st.markdown("# Ask the Model to Categorize Your Transaction")
st.markdown("""
            The model gives you two predictions:
            - The first is the category it thinks is most likely
            - The second is the category it thinks is second most likely
            """)

# Run model if user gave a valid API token
if api_token_success:

    # Create model client
    chat = ChatOpenAI(temperature=temperature, # want conservative temperature
                    openai_api_key=os.environ['API_TOKEN'],
                    # Select our model (best one ChatGPT has atm)
                    model_name = "gpt-4-1106-preview"
                    )
    # Run model once user clicks button
    st.markdown("**Shall we run the model now?**")
    run_model = st.checkbox('Yes please! :pray:', key="run_model")
    if run_model:
        
        # Give user a status bar
        with st.spinner(text="Your model is running...⏳"):

            # Get response for a given row
            response = chat(
                formatted_prompt
            )
            # Store predictions
            # Some hacky string processing necessary
            response_df = pd.DataFrame({"Category Prediction": response.content.split("\n")[0].strip().split(": ")[1],
                                        "Backup Prediction": response.content.split("\n")[1].strip().split(": ")[1]},
                                        index = [0])

            st.dataframe(response_df)

else:
    st.error("Please enter a valid API token to proceed!")

# # Get all responses - super slow with API
# all_responses = pd.DataFrame()

# for i, row in df.iterrows():
#     # Get in dict format
#     row = row.to_dict()

#     # get a chat completion from the formatted messages
#     response = chat(
#         chat_prompt.format_prompt(
#             counterparty_name=row["COUNTERPARTY_NAME"],
#             mcc_code=row["MCC_CODE"],
#             operation_type=row["OPERATION_TYPE"],
#             avg_spend_eur=row["AVG_SPEND_EUR"]
#         ).to_messages()
#     )
#     # Store predictions
#     # Some hacky string processing necessary
#     response_df = pd.DataFrame({"Category Prediction": response.content.split("\n")[0].strip().split(": ")[1],
#                                "Backup Prediction": response.content.split("\n")[1].strip().split(": ")[1]},
#                               index = [i])
#     all_responses = pd.concat([all_responses, response_df])
#     print(i)

    