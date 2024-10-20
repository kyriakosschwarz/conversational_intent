import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from business import *

# App Navigation using streamlit_option_menu
with st.sidebar:
    app_mode = option_menu(
        "Navigation", ["About", "Intent Prediction", "History", "Dashboard"],
        icons=["house", "activity", "airplane", "graph-up-arrow"],
        menu_icon="list", default_index=0
    )

if app_mode == "About":

    create_initial_table()

    st.title("About this app")
    st.markdown('**What can this app do?**')

    app_description = """
    This app is a conversational assistant. Here you can:
    1. Get an intent based on an user utterance.
    2. Display the history of predictions.
    3. View a dashboard with statistics over the predictions.
    """
    st.info(app_description)
    
    st.markdown('**How to use the app?**')

    app_usage = """
    To engage with the app use the navigation menu on the left.
    """
    st.warning(app_usage)


elif app_mode == "Intent Prediction":
    user_input = st.text_input("Enter user utterance:")
    if st.button("Get intent"):
        handle_get_intent_button(user_input)

elif app_mode == "History":
    st.title('Predictions History')

    if st.button("Show Last Entries"):
        handle_last_entries_button()

elif app_mode == "Dashboard":
    st.title('Intent Analysis Dashboard')
    #df = load_data()
    df = get_dashboard_data()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Probability Distribution Histogram')
        hist_fig = plot_probability_hist(df)
        if hist_fig:
            st.pyplot(hist_fig)

    with col2:
        st.subheader('Average Probability by Intent')
        avg_prob_fig = plot_avg_proba(df)
        if avg_prob_fig:
            st.pyplot(avg_prob_fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Word Cloud: High Probability Inputs (> 0.5)')
        high_prob_text = ' '.join(df[df['probability'] > 0.5]['input_text'])
        high_prob_fig = get_wordcloud(high_prob_text, 'High Probability Inputs')
        if high_prob_fig:
            st.pyplot(high_prob_fig)

    with col4:
        st.subheader('Word Cloud: Low Probability Inputs (<= 0.5)')
        low_prob_text = ' '.join(df[df['probability'] <= 0.5]['input_text'])
        low_prob_fig = get_wordcloud(low_prob_text, 'Low Probability Inputs')
        if low_prob_fig:
            st.pyplot(low_prob_fig)

