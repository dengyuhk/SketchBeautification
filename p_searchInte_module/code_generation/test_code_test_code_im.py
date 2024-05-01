import os
import pickle
from pathlib import Path



code_book_path= os.path.join('part_retrieval', 'code_books', 'vase_search_latent.pkl')
if Path(code_book_path).is_file():
    with Path(code_book_path).open('rb') as fh:
        code_books = pickle.load(fh)

code_im_book_path=os.path.join('part_retrieval', 'code_books', 'vase_search_latent_im.pkl')
if Path(code_im_book_path).is_file():
    with Path(code_im_book_path).open('rb') as fl:
        code_im_books = pickle.load(fl)
for partid in range(4):
    test=code_books[partid]['latent']
    test1=code_im_books[partid]['latent']
    a = test[0] - test1[0]
    print()

print()