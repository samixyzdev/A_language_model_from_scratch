from typing import Tuple, Dict, List, Iterable, Iterator
import regex as re
import pickle

class tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merge: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merge = merge
        self.special_tokens = special_tokens
        self.next_id = len(vocab)
        self.vocab_reverse = {}
        for key, val in self.vocab.items():
            self.vocab_reverse[val] = key
        for special_token in special_tokens:
            encoded_special_token = special_token.encode()
            if encoded_special_token not in self.vocab_reverse:
                self.vocab[self.next_id] = encoded_special_token
                self.vocab_reverse[encoded_special_token] = self.next_id
                self.next_id += 1
        # might need to add special_tokens that are not inside the vocab before doing reverse
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens: List[str] = None) -> 'tokenizer':
        loaded_vocab = {}
        loaded_merge = []
        with open(vocab_filepath, 'rb') as f:
            loaded_vocab = pickle.load(f)
        with open(merge_filepath, 'rb') as f:
            loaded_merge = pickle.load(f)
        return cls(loaded_vocab, loaded_merge, special_tokens)
    
    """
    append(element): Adds a single item to the end of a list. If the item is a list itself, it's added as a nested list.
    extend(iterable): Adds all individual items from an iterable (like another list, tuple, or string) to the end of the lis
    """
    
    def encode(self, text: str) -> List[int]:



    def encode_word(self, word: str) -> List[int]:
        token = List(word.encode())
        for merge_left, merge_right in self.merge:
            token = self.apply_merge_to_token(token, merge_left, merge_right)
        return token


    def apply_merge_to_token(self, token: List[int], merge_left: bytes, merge_right: bytes) -> List[int]:
        if len(token) == 1:
            return token
        new_token = []
        for i in range (len(token) - 1):
            if token[i] == self.vocab_reverse(merge_left) and token[i + 1] == self.vocab_reverse(merge_right):
                new_token.append(self.vocab_reverse[merge_left + merge_right])
                i += 1
            else:
                new_token.append(token[i])
        return new_token

    def encode_iterable(self, iterabel: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: List[int]) -> str:
        pass

            

