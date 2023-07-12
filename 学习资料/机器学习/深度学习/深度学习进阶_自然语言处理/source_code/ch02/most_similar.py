# coding: utf-8
import sys
# sys.path.append('..')
sys.path.append(r'D:\机器学习\深度学习\深度学习进阶_自然语言处理\source_code')
from common.util import preprocess, create_co_matrix, most_similar


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)
