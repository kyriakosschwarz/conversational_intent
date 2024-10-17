import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Function to create probability distribution histogram
def plot_probability_histogram(data):
    """
    Create a probability distribution histogram.

    This function generates a histogram plot of the probability distribution
    using the provided data. It uses seaborn's histplot function with a KDE overlay.

    Args:
        data (pandas.DataFrame): The data containing a 'probability' column.

    Returns:
        matplotlib.figure.Figure or None: The generated histogram plot, or None if there's not enough data.

    Note:
        If the input data is empty, a warning is displayed and the function returns None.
    """
    if data.shape[0] == 0:
        st.warning("Not enough data to plot Probability Distribution Histogram.")
        return None
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='probability', kde=True, ax=ax)
    ax.set_title('Probability Distribution Histogram')
    return fig

# Function to create average probability by intent bar chart
def plot_avg_probability_by_intent(data):
    """
    Create a bar chart of average probability by intent.

    This function calculates the mean probability for each intent and creates
    a bar chart to visualize these averages.

    Args:
        data (pandas.DataFrame): The data containing 'intent' and 'probability' columns.

    Returns:
        matplotlib.figure.Figure or None: The generated bar chart, or None if there's not enough data.

    Note:
        If the input data is empty or if no averages can be calculated, a warning is displayed
        and the function returns None.
    """
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
    """
    Create a word cloud visualization.

    This function generates a word cloud image from the provided text.

    Args:
        text (str): The text to generate the word cloud from.
        title (str): The title for the word cloud visualization.

    Returns:
        matplotlib.figure.Figure or None: The generated word cloud plot, or None if there's not enough data.

    Note:
        If the input text is empty or only contains whitespace, a warning is displayed
        and the function returns None.
    """
    if len(text.strip()) == 0:
        st.warning(f"Not enough data to create the Word Cloud for {title}.")
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig
