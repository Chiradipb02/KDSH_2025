from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
import time
from math import ceil
from evidence_checker_agent import EvidenceCheckerAgent
from methodology_assessment import methodoloy_assessment_agent
from content_evaluation_agent import content_eval_agent
from argument_evaluation_agent import CoherenceAgent

class aggregate_evaluation_agent:
    def __init__(self, model, api_key):
        self.evidence_checker = EvidenceCheckerAgent(model, api_key)
        self.methodology_assessment = methodoloy_assessment_agent(model, api_key)
        self.content_evaluation = content_eval_agent(model, api_key)
        self.coherence_agent = CoherenceAgent(model, api_key)
        self.system_prompt = '''You are an aggregate evaluation agent designed to compile and process scores from four distinct agents who assess a research paper from various perspectives. Each agent may provide either a single overall score or multiple scores for different areas of the paper. Your task is to compute a final aggregated score for the paper using the following process:

        Extract Scores from Each Agent:

        Check each agent's evaluation for an overall score out of 10.
        If an agent provides a single overall score, use it as is.
        If an agent provides multiple scores for different areas, calculate the mean of these scores.
        Calculate the Final Score:

        After obtaining one score from each of the four agents, calculate the mean of these four scores to determine the final score for the research paper.
        Ensure the final score is presented as a single number rounded to two decimal places.'''
                
        self.eval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "user", 
                    "This is the evaluation of the paper by content_evaluation_agent:\n\n {content_evaluation_response}.\n\n"
                    "This is the evaluation of the paper by methodology_assessment_agent:\n\n {methodology_assessment_response}.\n\n"
                    "This is the evaluation of the paper by argument_evaluation_agent:\n\n {argument_evaluation_response}.\n\n"
                    "This is the evaluation of the paper by evidence_checker_agent:\n\n {evidence_checker_response}.\n\n"
                    "now please aggregate the scores and provide the final mean score out of 10."
                )
            ]
        )
        self.aggregate_agent = self.eval_prompt | ChatGroq(model=model, api_key=api_key) | StrOutputParser()
        
    def aggregate_scores(self, content):
        content_evaluations = self.content_evaluation.evaluate_content(content)
        print("content evaluations: ", content_evaluations)
        methodology_evaluations = self.methodology_assessment.evaluate_content(content)
        print("methodology evaluations: ", methodology_evaluations)
        argument_evaluations = self.coherence_agent.evaluate_coherence(content)
        print("argument evaluations: ", argument_evaluations)
        evidence_checker_evaluations = self.evidence_checker.evaluate_coherence(content)   
        print("evidence checker evaluations: ", evidence_checker_evaluations) 
        # final_response = []
        # for i, chunk in enumerate(content):
        #     print(f"Evaluating chunk {i + 1}/{len(content)}...")
        #     evaluation = self.aggregate_agent.invoke({
        #         "content_evaluation_response": content_evaluations[i],
        #         "methodology_assessment_response": methodology_evaluations[i],
        #         "argument_evaluation_response": argument_evaluations[i],
        #         "evidence_checker_response": evidence_checker_evaluations[i]
        #     })
        #     final_response.append(evaluation) 
             
        final_response = self.aggregate_agent.invoke({
            "content_evaluation_response": content_evaluations,
            "methodology_assessment_response": methodology_evaluations,
            "argument_evaluation_response": argument_evaluations,
            "evidence_checker_response": evidence_checker_evaluations   
        })
        
        return final_response
    
    
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
    # print(type(headings_content))
    model="llama3-70b-8192"
    api_key="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"
    
    aggregate_agent = aggregate_evaluation_agent(model, api_key)
    final_response = aggregate_agent.aggregate_scores(headings_content)
    with open("final_response.txt", "w") as f:
        f.write(str(final_response))
    print("final response saved to 'final_response.txt'")

 
    
    
        
