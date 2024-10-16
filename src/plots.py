import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Function to create probability distribution histogram
def plot_probability_histogram(data):
    if data.shape[0] == 0:
        st.warning("Not enough data to plot Probability Distribution Histogram.")
        return None
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='probability', kde=True, ax=ax)
    ax.set_title('Probability Distribution Histogram')
    return fig

# Function to create average probability by intent bar chart
def plot_avg_probability_by_intent(data):
    if data.shape[0] == 0:
        st.warning("Not enough data to plot Average Probability by Intent.")
        return None
    avg_prob = data.groupby('intent')['probability'].mean().sort_values(ascending=False)
    if avg_prob.shape[0] == 0:
        st.warning("Not enough data to calculate Average Probability by Intent.")
        return None
    fig, ax = plt.subplots()
    sns.barplot(x=avg_prob.index, y=avg_prob.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Average Probability by Intent')
    plt.tight_layout()
    return fig

# Function to create word cloud
def create_wordcloud(text, title):
    if len(text.strip()) == 0:
        st.warning(f"Not enough data to create the Word Cloud for {title}.")
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig
