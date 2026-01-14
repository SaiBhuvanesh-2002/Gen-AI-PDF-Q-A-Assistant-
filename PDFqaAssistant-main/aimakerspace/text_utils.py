from typing import List, Dict, Any, Union
import fitz  # PyMuPDF
import os


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            content = f.read()
            self.documents.append(
                Document(page_content=content, metadata={"source": self.path})
            )

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding=self.encoding) as f:
                        content = f.read()
                        self.documents.append(
                            Document(page_content=content, metadata={"source": file_path})
                        )

    def load_documents(self):
        self.load()
        return self.documents


class PDFLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        try:
            doc = fitz.open(self.path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    self.documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": self.path, "page": page_num + 1},
                        )
                    )
            doc.close()
        except Exception as e:
            print(f"Error loading PDF {self.path}: {e}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    try:
                        doc = fitz.open(file_path)
                        for page_num, page in enumerate(doc):
                            text = page.get_text()
                            if text.strip():
                                self.documents.append(
                                    Document(
                                        page_content=text,
                                        metadata={
                                            "source": file_path,
                                            "page": page_num + 1,
                                        },
                                    )
                                )
                        doc.close()
                    except Exception as e:
                        print(f"Error loading PDF {file_path}: {e}")

    def load_documents(self):
        self.load()
        return self.documents


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", " ", ""],
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        separator = self.separators[-1]
        for sep in self.separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
        
        # If no separator found (unlikely as "" is fallback), split by chars is handled by recursion bottoming out
        
        splits = text.split(separator) if separator else list(text)
        
        # Now recombine
        current_chunk = []
        current_length = 0
        
        for split in splits:
            if separator: 
                split_len = len(split) + len(separator) # Approximation, varies for last element
            else: 
                split_len = len(split)

            if current_length + split_len > self.chunk_size:
                if current_length > 0:
                     # Join and add current chunk
                    doc_chunk = (separator if separator else "").join(current_chunk)
                    final_chunks.append(doc_chunk)
                    
                    # Reset
                    # Handle overlap - simplistic approach: keep last few items that fit in overlap
                    # For a true recursive splitter, we might re-split the large blob.
                    # Here we follow a simpler accumulation strategy similar to LangChain's basic logic.
                    
                    overlap_len = 0
                    new_chunk = []
                    # Backtrack to fill overlap
                    for item in reversed(current_chunk):
                         item_len = len(item) + (len(separator) if separator else 0)
                         if overlap_len + item_len <= self.chunk_overlap:
                             new_chunk.insert(0, item)
                             overlap_len += item_len
                         else:
                             break
                    current_chunk = new_chunk
                    current_length = overlap_len
                
            current_chunk.append(split)
            current_length += split_len
            
        if current_chunk:
             final_chunks.append((separator if separator else "").join(current_chunk))
             
        # Verification: If any chunk is still too large, recursively split it with next separator
        # This is the "Recursive" part.
        
        output_chunks = []
        for chunk in final_chunks:
            if len(chunk) > self.chunk_size:
                # Find index of current separator
                try:
                    next_sep_index = self.separators.index(separator) + 1
                    if next_sep_index < len(self.separators):
                         # Recursively split this chunk with the next separator
                         sub_splitter = RecursiveCharacterTextSplitter(
                             chunk_size=self.chunk_size,
                             chunk_overlap=self.chunk_overlap,
                             separators=self.separators[next_sep_index:]
                         )
                         output_chunks.extend(sub_splitter.split_text(chunk))
                    else:
                        # No more separators, hard cut
                        output_chunks.extend(self._hard_cut(chunk))
                except ValueError:
                     # This shouldn't happen if logic is sound
                     output_chunks.extend(self._hard_cut(chunk))
            else:
                output_chunks.append(chunk)

        return output_chunks

    def _hard_cut(self, text: str) -> List[str]:
        # Fallback for when no separators work (really long string of garbage)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_documents = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                if chunk.strip():
                    final_documents.append(
                        Document(page_content=chunk, metadata=doc.metadata.copy())
                    )
        return final_documents

if __name__ == "__main__":
    # Test
    text = "Paragraph 1.\n\nParagraph 2 is longer.\nAnd has multiple lines.\n\nParagraph 3."
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    chunks = splitter.split_text(text)
    print("Chunks:", chunks)
