import os
import json
import re

def sent_tokenize(text):
    """
    A simple sentence tokenizer using regular expressions.
    Splits on . ! ? followed by a space and a capital letter or end of string.
    """
    # Normalize whitespace: replace newlines and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if not text:
        return []
    # Add a trailing space to help the pattern match the last sentence
    text = text + ' '
    # Pattern: any characters, then sentence-ending punctuation, then space,
    # then either a capital letter or the end of the string.
    sentences = re.findall(r'(.*?[.!?])\s+(?=[A-Z]|$)', text)
    # If no sentences were found (e.g., text without punctuation), treat the whole text as one sentence
    if not sentences:
        sentences = [text.strip()]
    return sentences

def chunk_sentences(sentences, target_words=250):
    """
    Chunk a list of sentences into overlapping chunks.
    Each chunk targets approximately `target_words` words.
    If the first sentence of a chunk already contains >= target_words,
    it becomes a chunk by itself (no overlap).
    Otherwise, sentences are added until adding the next sentence would exceed the target.
    That next sentence is then included (to complete it) and becomes the overlap for the next chunk.
    """
    chunks = []
    i = 0
    while i < len(sentences):
        # Case: the current sentence alone is already at least target_words
        if len(sentences[i].split()) >= target_words:
            chunks.append([sentences[i]])
            i += 1
            continue

        # Normal case: build a chunk starting at index i
        chunk_sents = []
        word_count = 0
        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_words = len(sent.split())

            # If we already have at least one sentence and adding this one would exceed the target,
            # we include it (to complete the sentence) and decide where the next chunk starts.
            if len(chunk_sents) > 0 and word_count + sent_words > target_words:
                chunk_sents.append(sent)
                word_count += sent_words
                # If there are more sentences after this one, overlap by starting the next chunk here
                if j + 1 < len(sentences):
                    i = j          # overlap: next chunk starts with this sentence
                else:
                    i = j + 1       # last sentence, no next chunk
                break
            else:
                # Add the sentence normally and continue
                chunk_sents.append(sent)
                word_count += sent_words
                j += 1
                # If we've consumed all sentences, set i to the end
                if j == len(sentences):
                    i = j
                    break

        chunks.append(chunk_sents)

    return chunks

def main():
    data_folder = "rich-data"
    if not os.path.isdir(data_folder):
        print(f"Error: folder '{data_folder}' does not exist.")
        return

    all_chunks = []
    # Process every .txt file in the folder
    for filename in os.listdir(data_folder):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(data_folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        sentences = sent_tokenize(text)
        if not sentences:
            continue

        chunks = chunk_sentences(sentences, target_words=250)

        for idx, chunk_sents in enumerate(chunks):
            chunk_text = ' '.join(chunk_sents)
            word_count = sum(len(s.split()) for s in chunk_sents)
            all_chunks.append({
                "source": filename,
                "chunk_index": idx,
                "text": chunk_text,
                "word_count": word_count
            })

    # Write all chunks to a JSON file
    output_file = "chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunking complete. {len(all_chunks)} chunks saved to {output_file}")

if __name__ == "__main__":
    main()