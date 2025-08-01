from typing import Tuple, Dict, List, Iterable, Iterator
import regex as re
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merge: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merge = merge
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key = len, reverse = True)
        escaped_special_tokens = [re.escape(special_token) for special_token in self.special_tokens]
        if escaped_special_tokens:
            self.special_token_pattern = '|'.join(escaped_special_tokens)
        else:
            self.special_token_pattern = None
        self.combined_split_pattern = re.compile(PAT, re.UNICODE)

        self.next_id = len(vocab)
        self.vocab_reverse = {}
        for key, val in self.vocab.items():
            self.vocab_reverse[val] = key
        for special_token in self.special_tokens:
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
        encoded_text = []
        if self.special_tokens:
            text_segments = re.split(f'({self.special_token_pattern})', text)
            for segment in text_segments:
                if not segment:
                    continue
                if segment in self.special_tokens:
                    encoded_text.append(self.vocab_reverse[segment.encode()])
                else:
                    split_segments = re.findall(self.combined_split_pattern, segment)
                    for split_segment in split_segments:
                        if split_segment:
                            encoded_text.extend(self.encode_word(split_segment))
        else:
            text_segments = re.findall(self.combined_split_pattern, text)
            for segment in text_segments:
                if segment:  
                    encoded_text.extend(self.encode_word(segment))
        return encoded_text
    

    def encode_word(self, word: str) -> List[int]:
        word_bytes = word.encode()
        token = []
        if word_bytes in self.vocab_reverse:
            token = [self.vocab_reverse[word_bytes]]
        else:
            for byte in word_bytes:
                token.append(self.vocab_reverse[bytes([byte])])
        for merge_left, merge_right in self.merge:
            token = self.apply_merge_to_token(token, merge_left, merge_right)
        return token


    def apply_merge_to_token(self, token: List[int], merge_left: bytes, merge_right: bytes) -> List[int]:
        if len(token) <= 1:
            return token
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == self.vocab_reverse[merge_left] and token[i + 1] == self.vocab_reverse[merge_right]:
                new_token.append(self.vocab_reverse[merge_left + merge_right])
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        return new_token

    def encode_iterable(self, iterabel: Iterable[str]) -> Iterator[int]:
        for text in iterabel:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        ids_to_bytes = []
        for id in ids:
            ids_to_bytes.append(self.vocab[id])
        full_seq = b"".join(ids_to_bytes)
        return full_seq.decode()

            

