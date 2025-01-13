from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
class methodoloy_assessment_agent:
    def __init__(self, model, api_key):
        self.llm=ChatGroq(model=model, api_key=api_key)
        self.system_prompt= '''You are an Appropriateness analyzer of research papers.
        Analyze the appropriateness and justification of the methodologies used in this research paper.
         Discuss whether the methods align with the research objectives, and evaluate their strengths and weaknesses in addressing the research questions.
         Provide an assessment of the validity, reliability, and generalizability of the chosen methodologies.
         Include specific examples from the paper to support your analysis.
         Specifically:
         1. Mention if the chosen methods directly support the research goals and questions.
         2. Whether the paper has Provided evidence from past research or pilot studies that supports the selection of the methodology.
         3. Whether the methods are practically implemented within available resources.
         4. Whether the paper used a lot of terms and concepts are used throughout without proper explanation.

         Positive score is given if the above first 3 points are supported, last one is negative point.Score the paper out of 10. 
         Give the whole response in 50 words\n\n'''

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [

                    ("system", self.system_prompt),


                    ( "user", "This is the content of the research paper:\n\n {content}.\n\n Please evaluate it and rate the overall quality of paper out of 10.""Give the whole response in 50 words\n\n",)
            ,
            ]
        )
        self.content_agent=self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_content(self, content: str) -> str:
        evaluations = []
        for i, chunk in enumerate(content):
            print(f"Evaluating chunk {i + 1}/{len(content)}...")
            evaluation = self.content_agent.invoke({"content": chunk["content"]})
            evaluations.append(f"Evaluation for {chunk['heading']}:\n{evaluation}\n")
        return "\n".join(evaluations)



from math import ceil

def split_text(text, max_tokens=3000):
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


def extract_and_chunk_paper(pdf_path, max_tokens=3000):
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
    content_evaluator = methodoloy_assessment_agent(model, api_key)
    evaluation = content_evaluator.evaluate_content(headings_content)
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write(evaluation)
    
    print("Evaluation completed. Results saved to 'evaluation.txt'.")


