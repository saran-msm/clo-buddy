import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import time
import backoff
import PyPDF2
from io import BytesIO
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from .vector_store import VectorStore

class DocumentProcessor:
    def __init__(self):
        load_dotenv(override=True)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        try:
            self.model_name = "google/flan-t5-base"
            print(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Loading model...")
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Loading embedding model {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            self.model.to(self.device)
            
            # Model parameters
            self.block_size = 256
            self.max_input_length = 256
            self.max_output_length = 128
            self.min_output_length = 32
            self.num_beams = 2
            self.timeout = 30
            self.executor = ThreadPoolExecutor(max_workers=3)
            
            # Updated system prompts
            self.input_prompt = """
            You will be provided with a legal document. Your task is to process the document and provide the following in a structured format:
            1. **Concise Summary** (200-300 words): Summarize the document without repeating information. Focus on the most important aspects, and ensure clarity and conciseness.
            2. **Key Highlights/Clauses**: Extract and list the most significant clauses or sections of the document (e.g., obligations, rights, penalties, deadlines). Avoid rephrasing the summary.
            3. **Relevant Legal References**: Identify and list any statutes, case laws, or legal references mentioned in the document.
            4. **Actionable Insights**: If applicable, provide next steps or actionable items based on the document's contents. These should be clear and specific.

            Document: {text}"""
            
            self.output_format = """
            **Summary:**
            {summary}

            **Key Highlights/Clauses:**
            1. {highlights}

            **Legal References:**
            - {references}

            **Actionable Insights:**
            1. {insights}

            *Note: This summary is for informational purposes only and should not be considered as legal advice.*
            """
            
            self.vector_store = VectorStore()
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def process_with_timeout(self, func, *args, **kwargs):
        """Process with timeout using ThreadPoolExecutor"""
        try:
            future = self.executor.submit(func, *args, **kwargs)
            return future.result(timeout=self.timeout)
        except TimeoutError:
            raise TimeoutError("Processing took too long")

    def extract_text(self, file_obj, filename: str) -> str:
        """Extract text with optimized processing"""
        try:
            if filename.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_obj.read()))
                text = ""
                chunk_size = 5
                for i in range(0, len(pdf_reader.pages), chunk_size):
                    chunk = pdf_reader.pages[i:i + chunk_size]
                    for page in chunk:
                        text += page.extract_text() + "\n"
                return text.strip()
            else:
                return file_obj.read().decode('utf-8')
                    
        except Exception as e:
            print(f"Error extracting text from {filename}: {str(e)}")
            raise ValueError(f"Failed to extract text from {filename}: {str(e)}")

    def chunk_text(self, text: str, max_chunk_tokens: int = 500) -> List[str]:
        """Split text into smaller chunks"""
        try:
            # Reduce chunk size to avoid token length errors
            chunks = []
            sentences = text.split('. ')
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                # Add period back to sentence
                sentence = sentence.strip() + '. '
                sentence_tokens = len(self.tokenizer.encode(sentence))
                
                if current_length + sentence_tokens > max_chunk_tokens:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            print(f"Error chunking text: {str(e)}")
            raise

    def generate_summary(self, text: str) -> str:
        """Generate summary from chunks"""
        try:
            # Split into smaller chunks
            chunks = self.chunk_text(text)
            summaries = []
            
            # Process each chunk
            for chunk in chunks[:5]:  # Limit to first 5 chunks for summary
                formatted_input = f"Summarize this legal text: {chunk}"
                
                inputs = self.tokenizer(
                    formatted_input,
                    max_length=500,  # Reduced from 512
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=100,  # Reduced output length
                        min_length=30,
                        num_beams=2,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                
                summary = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                summaries.append(summary)
            
            # Combine summaries
            final_summary = ' '.join(summaries)
            return final_summary[:500]  # Limit final summary length
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            raise

    def extract_highlights(self, text: str) -> Dict[str, List[str]]:
        """Extract highlights from chunks"""
        try:
            # Print input text length
            print(f"\nInput text length: {len(text.split())} words")
            
            # Strictly limit input text
            words = text.split()[:50]
            truncated_text = ' '.join(words)
            print(f"Truncated text length: {len(words)} words")
            
            def generate_section(prompt, section_name):
                try:
                    # Create new model instance
                    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(self.device)
                    
                    # Print section start
                    print(f"\nGenerating {section_name}...")
                    
                    # Create input text
                    input_text = f"{prompt}:\n{truncated_text}"
                    print(f"Prompt: {prompt}")
                    
                    # Tokenize
                    tokens = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=500,
                        return_tensors="pt"
                    )
                    
                    # Generate
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=tokens["input_ids"].to(self.device),
                            attention_mask=tokens["attention_mask"].to(self.device),
                            max_length=50,
                            min_length=10,
                            num_beams=2,
                            early_stopping=True
                        )
                    
                    # Decode result
                    result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    print(f"\nRaw {section_name} output:")
                    print(result)
                    
                    # Split into lines
                    lines = [line.strip() for line in result.split('\n') if line.strip()]
                    print(f"\nProcessed {section_name}:")
                    for line in lines:
                        print(f"- {line}")
                    
                    # Cleanup
                    del model
                    torch.cuda.empty_cache()
                    
                    return lines
                    
                except Exception as e:
                    print(f"Error in {section_name}: {str(e)}")
                    return []
            
            # Generate highlights
            print("\n=== Generating Key Highlights ===")
            highlights = generate_section("List exactly 3 key points", "Key Highlights")
            
            # Generate insights
            print("\n=== Generating Actionable Insights ===")
            insights = generate_section("List exactly 2 action items", "Actionable Insights")
            
            # Format and print final outputs
            print("\n=== Final Formatted Outputs ===")
            
            if not highlights or len(highlights) < 3:
                print("\nUsing default highlights...")
                highlights = [
                    "Document review required",
                    "Content analysis needed",
                    "Key points to be extracted"
                ]
            
            if not insights or len(insights) < 2:
                print("\nUsing default insights...")
                insights = [
                    "Conduct detailed review",
                    "Consult with stakeholders"
                ]
            
            # Format with numbers
            highlights = [f"{i+1}. {h}" for i, h in enumerate(highlights[:3])]
            insights = [f"{i+1}. {ins}" for i, ins in enumerate(insights[:2])]
            
            print("\nFinal Key Highlights:")
            for highlight in highlights:
                print(highlight)
            
            print("\nFinal Actionable Insights:")
            for insight in insights:
                print(insight)
            
            return {
                'highlights': highlights,
                'references': ["- Analysis completed"],
                'insights': insights,
                'total_highlights': len(highlights)
            }
            
        except Exception as e:
            print(f"\nError in extract_highlights: {str(e)}")
            default_response = {
                'highlights': [
                    "1. Document analysis required",
                    "2. Review content",
                    "3. Extract key points"
                ],
                'references': ["- Review needed"],
                'insights': [
                    "1. Perform detailed review",
                    "2. Identify action items"
                ],
                'total_highlights': 3
            }
            print("\nReturning default response:")
            print("\nDefault Highlights:")
            for h in default_response['highlights']:
                print(h)
            print("\nDefault Insights:")
            for i in default_response['insights']:
                print(i)
            return default_response 

    def process_document(self, file_obj, filename: str) -> Dict:
        """
        Process document and store in vector database
        """
        try:
            # Extract text
            text = self.extract_text(file_obj, filename)
            
            # Generate document ID (you might want to use a more sophisticated method)
            document_id = f"doc_{int(time.time())}"
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Store in vector database
            self.vector_store.add_document(
                document_id=document_id,
                text_chunks=chunks,
                metadata={
                    "filename": filename,
                    "timestamp": time.time(),
                    "total_chunks": len(chunks)
                }
            )
            
            # Generate summary
            summary = self.generate_summary(text)
            
            # Extract highlights
            highlights_data = self.extract_highlights(text)
            
            return {
                "document_id": document_id,
                "summary": summary,
                "highlights": highlights_data["highlights"],
                "insights": highlights_data["insights"],
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            print(f"Error processing document: {e}")
            raise

    def search_similar_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for similar content in stored documents
        """
        try:
            return self.vector_store.search(query, n_results)
        except Exception as e:
            print(f"Error searching similar content: {e}")
            raise 