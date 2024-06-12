import pandas as pd
import numpy as np
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')


# streamlit app
st.set_page_config(page_title="Pelabelan Data app", page_icon="Icon/management.png")

# Streamlit UI
st.title("Pelabelan Data")

# Upload file CSV, XLSX, atau TXT
uploaded_tweet_file = st.file_uploader("Upload File Komentar", type=["xlsx"])

if uploaded_tweet_file:
    if uploaded_tweet_file.type == "csv":
        dt_tweet = pd.read_csv(uploaded_tweet_file, encoding='utf-8')
    elif uploaded_tweet_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dt_tweet = pd.read_excel(uploaded_tweet_file)
    elif uploaded_tweet_file.type == "text/plain":
        # Jika file adalah TXT
        # Gunakan readlines() untuk membaca baris per baris
        data = uploaded_tweet_file.getvalue().decode("utf-8")
        dt_tweet = pd.DataFrame({"komentar": data.split("\n")})

    st.write("Uploaded Tweet Data:")
    st.dataframe(dt_tweet, height=300)

    # Case folding
    dt_tweet['case_folding'] = dt_tweet['komentar'].str.lower()

    # Function to remove special characters
    def remove_tweet_special(text):
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub(r"([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
        text = re.sub(r"[_?]", " ", text)
        return text.replace("http://", " ").replace("https://", " ")

    dt_tweet['remove_punctuation'] = dt_tweet['case_folding'].apply(remove_tweet_special)

    def handling_remove_punctuation(text):
    # Menghapus tanda baca
        return re.sub(r'[^\w\s]', '', text)

    dt_tweet['remove_punctuation'] = dt_tweet['remove_punctuation'].apply(handling_remove_punctuation)
    
    # Function to remove numbers
    def remove_number(text):
        return re.sub(r"\d+", " ", text)

    dt_tweet['remove_number'] = dt_tweet['remove_punctuation'].apply(remove_number)

    def handling_remove_single_char(text):
    # Menghapus karakter tunggal
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    dt_tweet['remove_single_char'] = dt_tweet['remove_number'].apply(handling_remove_single_char)

    # Tokenization
    def tokenize_word(text):
        return word_tokenize(text)

    dt_tweet['tweet_tokens'] = dt_tweet['remove_single_char'].apply(tokenize_word)

    # Remove stopwords
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                           'kalo', 'amp', 'biar', 'bikin', 'bilang',
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah'])

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    dt_tweet['tweet_tokens_WSW'] = dt_tweet['tweet_tokens'].apply(stopwords_removal)

    # Display preprocessed data
    st.write("Preprocessed Tweet Data:")
    st.dataframe(dt_tweet, height=300)

    # Upload lexicon files
    uploaded_positive_lexicon = st.file_uploader("Upload CSV file containing positive lexicon", type=["csv"])
    uploaded_negative_lexicon = st.file_uploader("Upload CSV file containing negative lexicon", type=["csv"])

    if uploaded_positive_lexicon and uploaded_negative_lexicon:
        # Load positive lexicon
        lexicon_positive = dict()
        with uploaded_positive_lexicon as csvfile:
            reader = pd.read_csv(csvfile)
            for index, row in reader.iterrows():
                lexicon_positive[row['word']] = int(row['weight'])

        # Load negative lexicon
        lexicon_negative = dict()
        with uploaded_negative_lexicon as csvfile:
            reader = pd.read_csv(csvfile)
            for index, row in reader.iterrows():
                lexicon_negative[row['word']] = int(row['weight'])

        def sentiment_analysis_lexicon_indonesia(tokens):
            score = 0
            for word in tokens:
                word = word.lower()
                if word in lexicon_positive:
                    score += lexicon_positive[word]
                if word in lexicon_negative:
                    score += lexicon_negative[word]
            polarity = ''
            if score > 0:
                polarity = 'positive'
            elif score < 0:
                polarity = 'negative'
            else:
                polarity = 'neutral'
            return score, polarity

        results = dt_tweet['tweet_tokens_WSW'].apply(sentiment_analysis_lexicon_indonesia)
        results = list(zip(*results))
        dt_tweet['polarity_score'] = results[0]
        dt_tweet['polarity'] = results[1]

        st.write("Processed Tweet Data with Sentiment Analysis:")
        st.dataframe(dt_tweet, height=300)

        # Convert DataFrame to CSV
        csv = dt_tweet.to_csv(index=False)

        # Convert CSV string to bytes
        csv_bytes = io.BytesIO(csv.encode('utf-8'))

        # Create a download button
        st.download_button(
            label="Download Pelabelan Data as CSV",
            data=csv_bytes,
            file_name="Pelabelan Data.csv",
            mime="text/csv"
        )

        # Display polarity count
        st.write("Sentiment Polarity Distribution:")
        st.write(dt_tweet['polarity'].value_counts())

        # Plot sentiment distribution
        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [count for count in dt_tweet['polarity'].value_counts()]
        labels = list(dt_tweet['polarity'].value_counts().index)
        explode = (0.1, 0, 0)
        ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 14})
        ax.set_title('Sentiment Polarity on Tweets Data', fontsize=16, pad=20)
        st.pyplot(fig)

        # Display positive tweets
        st.write("Positive Tweets:")
        positive_tweets = dt_tweet[dt_tweet['polarity'] == 'positive']
        positive_tweets = positive_tweets[['tweet_tokens_WSW', 'polarity_score', 'polarity']].sort_values(by='polarity_score', ascending=False).reset_index(drop=True)
        positive_tweets.index += 1
        st.write(positive_tweets)

        # Display negative tweets
        st.write("Negative Tweets:")
        negative_tweets = dt_tweet[dt_tweet['polarity'] == 'negative']
        negative_tweets = negative_tweets[['tweet_tokens_WSW', 'polarity_score', 'polarity']].sort_values(by='polarity_score', ascending=True).reset_index(drop=True)
        negative_tweets.index += 1
        st.write(negative_tweets)

        # Display neutral tweets
        st.write("Neutral Tweets:")
        neutral_tweets = dt_tweet[dt_tweet['polarity'] == 'neutral']
        neutral_tweets = neutral_tweets[['tweet_tokens_WSW', 'polarity_score', 'polarity']].sort_values(by='polarity_score', ascending=True).reset_index(drop=True)
        neutral_tweets.index += 1
        st.write(neutral_tweets)

