from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
from pdfminer.high_level import extract_text
from utils.llm_api import llm_api
class content_eval_agent:
    def __init__(self, model, api_key):
        self.llm=ChatGroq(model=model, api_key=api_key)
        self.system_prompt= "You are a content evaluation assistant for research papers. Your task is to assess whether the "
        "provided research paper includes all critical sections required for publication and if the content "
        "in each section is sufficiently detailed and relevant.\n\n"
        "Specifically:\n"
        "1. Ensure the paper includes the following sections: Abstract, Introduction, Methodology, Results, "
        "Discussion, and Conclusion.\n"
        "2. Highlight missing sections, if any.\n"
        "3. Identify sections with insufficient or irrelevant details.\n"
        "4. Provide actionable feedback on how the paper can be improved.\n"
        "5. Ensure your response is structured as a checklist followed by recommendations.\n\n"
        "6. Give a rating out of 10 for the overall quality of the paper.\n\n"
        "Return your evaluation in the following format:\n"
        "- Section: [Present/Missing/Insufficient]\n"
        "- Feedback: [Detailed feedback for improvement or acknowledgment if the section is good]\n"
        "- Rating: [Rating out of 10]\n"
        " Give the whole response in 50 words\n\n"
        
        self.eval_prompt = ChatPromptTemplate.from_messages(
            [
                
                    ("system", self.system_prompt),
                
                
                    ( "user", "This is the content of the research paper:\n\n {content}.\n\n Please evaluate it and rate the overall quality of paper out of 10.""Give the whole response in 50 words\n\n",)
            ,
            ]
        )
        self.content_agent=self.eval_prompt | (lambda prompt: llm_api(prompt, model="gpt", api_key=api_key)) | StrOutputParser()

    def evaluate_content(self, content: str) -> str:
        evaluations = []
        for i, chunk in enumerate(content):
            print(f"Evaluating chunk {i + 1}/{len(content)}...")
            evaluation = self.content_agent.invoke({"content": chunk["content"]})
            evaluations.append(f"Evaluation for {chunk['heading']}:\n{evaluation}\n")
        return "\n".join(evaluations)
    


from math import ceil

def split_text(text, max_tokens=5500):
    """
    Splits text into smaller chunks to meet token limit.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum tokens allowed in each chunk.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunk_size = max_tokens  # Approximation based on words
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def extract_and_chunk_paper(pdf_path, max_tokens=3500):
    """
    Extracts headings and content from a PDF and ensures they fit within token limits.

    Args:
        pdf_path (str): Path to the PDF file.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list: List of dictionaries with headings and corresponding content chunks.
    """
    doc = pymupdf.open(pdf_path)
    headings_content = {}
    current_heading = "General"
    content_buffer = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    font_size = line["spans"][0]["size"]

                    if font_size > 12:  # Treat larger font sizes as headings
                        if current_heading and content_buffer:
                            headings_content[current_heading] = content_buffer.strip()
                        current_heading = text
                        content_buffer = ""
                    else:
                        content_buffer += " " + text

    if current_heading and content_buffer:
        headings_content[current_heading] = content_buffer.strip()

    doc.close()

    # Chunk content under each heading
    chunked_output = []
    for heading, content in headings_content.items():
        chunks = split_text(content, max_tokens=max_tokens)
        for idx, chunk in enumerate(chunks):
            chunked_output.append({
                "heading": f"{heading} (Part {idx + 1})" if len(chunks) > 1 else heading,
                "content": chunk
            })

    return chunked_output




if __name__ == "__main__":
    pdf_path = "bad_paper_3.pdf"
    headings_content = extract_and_chunk_paper(pdf_path)
    model="llama3-70b-8192"
    api_key="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"
    content_evaluator = content_eval_agent(model, api_key)
    evaluation = content_evaluator.evaluate_content(headings_content)
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write(evaluation)
    
    print("Evaluation completed. Results saved to 'evaluation.txt'.")

