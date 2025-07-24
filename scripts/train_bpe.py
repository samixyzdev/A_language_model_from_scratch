import os
import regex as re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, BinaryIO
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

"""
bytes([65]) creates a bytes object from a list containing the integer 65. Since 65 is the ASCII code for 'A', this results in b'A'.
bytes(65) creates a bytes object of length 65, filled with zeros. This gives you b'\x00\x00\x00...' (65 zero bytes total).
"""

"""
Problem encountered, not sure if there are multiple special tokens , and the current function doesn't support multiple funcitons
problem resolved: only one special token exist
"""

def process_chunk(args: Tuple[str, int, int, List[str]]) -> Counter: # path, start_pos, end_pos, special_token

    filepath, start_pos, end_pos, special_tokens = args
    
    # Read the specific chunk from file (bytes start_pos to end_pos)

    """
    The f.seek(start_pos) line is absolutely essential. Here's why:

    Unpredictable File Pointer: When you open a file or if previous read/write operations occurred, the file's internal pointer (cursor) might not be at the start_pos you need for the current chunk. f.seek(start_pos) explicitly moves the pointer to that exact byte position.

    Ensuring Correct Chunk Reading: Each parallel process needs to read a specific, unique section of the file defined by its start_pos and end_pos. Without f.seek(), processes might start reading from the beginning of the file, or from where a previous read left off, leading to:

    Duplicate data: Multiple processes reading the same content.

    Missing data: Parts of the file being skipped.

    Incorrect results: Your final aggregated data would be wrong. 

    string.strip([chars])

    string: The string you want to strip.

    [chars]: An optional argument. If provided, it specifies the set of characters to be removed from the beginning and end of the string. If chars is not provided, strip() removes all types of whitespace characters by default.
    """
    with open(filepath, "rb") as f:
        f.seek(start_pos)
        chunk_bytes = f.read(end_pos - start_pos)
        # Decode bytes to string
    chunk_data = chunk_bytes.decode("utf-8", errors="ignore")
    # Remove/handle the special_token 
    escape_tokens = [re.escape(special_token) for special_token in special_tokens]
    escape_tokens_pattern = "|".join(escape_tokens)
    segments = re.split(escape_tokens_pattern, chunk_data, flags=re.U) 
    # Apply regex pre-tokenization using PAT
    total_token_frequency = Counter()
    for segment in segments:
        if not segment.strip():
            continue
        raw_tokens = re.findall(PAT, segment, re.U)
        processed_token = []
        for token in raw_tokens:
            token = token.strip().lower()
            if token:
                processed_token.append(token)
        total_token_frequency.update(processed_token)
    return total_token_frequency
            

def parallel_pretokenize(input_path: str, special_tokens: List[str], num_process: int = None) -> Counter:
    if num_process == None:
        num_process = cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, "<|endoftext|>".encode("utf-8"))
    chunk_args = []
    for i in range(len(boundaries) - 1):
        start_pos = boundaries[i]
        end_pos = boundaries[i+1]
        chunk_args.append((input_path, start_pos, end_pos, special_tokens))
    with Pool(num_process) as pool:
        chunk_results = pool.map(process_chunk, chunk_args) # a list of count obj
    total_frequency = Counter();
    for chunk_result in chunk_results:
        total_frequency.update(chunk_result)
    return total_frequency
    
def count_pairs(byte_tokens_frequency: Dict[Tuple, int]) -> Counter:
    pair_counts = Counter()
    for byte_tokens, frequency in byte_tokens_frequency.items():
        for i in range(len(byte_tokens) - 1):
            pair = (byte_tokens[i], byte_tokens[i+1])
            pair_counts[pair] += frequency
    return pair_counts

def find_best_pair(pairs: Counter) -> Tuple[int, int]:
    if not pairs:
        return None
    best_pair = max(pairs, key = lambda p: (pairs[p], -p[0], -p[1]))
    return best_pair
# REASONING:
# _find_best_pair function aims to select the most frequent byte pair for merging.
#
# It uses Python's built-in `max()` function with a `key` argument.
# When `key` is provided, `max()` determines the "largest" item based on the
# return value of the `key` function, not the item itself.
#
# The `lambda p: (pair_counts[p], -p[0], -p[1])` is an anonymous function
# that defines the sorting criteria:
#
# 1. `pair_counts[p]`: This is the primary sorting key. `max()` will prioritize
#    pairs with higher frequencies. (Highest count first)#
#
# 2. `-p[0]`: If frequencies are tied, this is the secondary sorting key.
#    `p` is the byte pair (e.g., (104, 101)). `p[0]` is the first byte.
#    By negating `p[0]`, we achieve lexicographical (ascending) order for the
#    first byte. `max()` will pick the *largest* value from the `key`'s output.
#    A smaller original `p[0]` results in a larger negative value (e.g., -10 is greater than -20),
#    causing `max` to select the pair with the *smaller* first byte.
#    (Smallest first byte value first)
#
# 3. `-p[1]`: If both frequency and the first byte are tied, this is the tertiary
#    sorting key. Similarly, by negating `p[1]`, we achieve lexicographical
#    (ascending) order for the second byte.
#    (Smallest second byte value first)
#
# The `max()` function iterates over the `keys` of `pair_counts` (which are `Tuple[int, int]`).
# It returns the *original key* (the `Tuple[int, int]`) that yields the
# "largest" comparison tuple defined by the `lambda` function.
# This ensures a deterministic and reproducible selection of the best pair,
# which is crucial for consistent BPE training results.

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[256 + i] = special_tokens[i]
    tokens_frequency = parallel_pretokenize(input_path, special_tokens)
    byte_tokens_frequency = {}
    for token_str, token_fq in tokens_frequency.items():
        byte_tokens_frequency[tuple(token_str.encode())] = token_fq

    """
    tuple(word_str.encode("utf-8")) will turn str into a tuple of bytes in utf-8 
    items() will retuen (key, value) of a dict / Counter
    """