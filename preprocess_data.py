import re
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import requests
import chardet
from bs4 import BeautifulSoup

#daughterâ€™s

URL = 'https://www.gutenberg.org/files/{id}/'


'''
This class is used to preprocess the text data from Gutenberg free books.
'''
class PreprocessData:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        self.corpus = ''

    def download_data(self, from_id=1, limit=100):
        assert from_id > 0, 'from_id should be greater than 0'
        assert from_id + limit < 10000, 'from_id + limit should be less than 1000'
        for i in tqdm(range(from_id, from_id + limit)):
            try:
                r = requests.get(URL.format(id=i))
                html = r.content.decode('utf-8')
                bs = BeautifulSoup(html, 'html.parser')
                for a_tag in bs.find_all('a'):
                    if a_tag.get('href').endswith('.txt'):
                        r = requests.get(URL.format(id=i) + a_tag.get('href'))  # url + name of the book file that represents like id.txt
                        text = r.content.decode('utf-8')
                        self.corpus += text
                        break
            except Exception as e:
                print('Error:', e)
                print('This is file was skipped')

    def tokenize(self, remove_stop_words=True):
        if not self.corpus:
            print('The corpus is empty')
            return

        self.corpus = self.corpus.lower()
        # remove non-alphabetic characters
        self.corpus = re.sub(r'[^a-z]', ' ', self.corpus)
        words = word_tokenize(self.corpus, language='english')
        # remove stop words
        if remove_stop_words:
            words = [word for word in words if word not in self.stop_words]
        return words


# 1: 14435, 10: 139796, 50: 945357, 100: 3346189
# TODO open question: stopwords helps to CBOW in Word2Vec?