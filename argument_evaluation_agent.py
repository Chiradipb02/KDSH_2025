from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf

class CoherenceAgent:
    """
    Evaluates the logical consistency, clarity, and organization of arguments in the given content.

    Args:
        model (str): The model to be used for evaluation.
        api_key (str): The API key for accessing the model.
    """
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt = (
            "You are an expert in academic writing and research evaluation. Your task is to assess the logical coherence, "
            "clarity, and organization of arguments in research papers.\n\n"
            "When evaluating, follow these guidelines:\n"
            "1. Logical Consistency:\n"
            "   - Check if the arguments follow a logical progression, with premises leading to conclusions.\n"
            "   - Identify contradictions, unsupported claims, or inconsistencies.\n"
            "2. Clarity:\n"
            "   - Ensure that the arguments are expressed clearly and precisely.\n"
            "   - Look for ambiguous, overly complex, or unclear language and suggest improvements.\n"
            "3. Organization:\n"
            "   - Evaluate the structure of the arguments to ensure they are cohesive and logically organized.\n"
            "   - Check if sections are well-connected and flow naturally from one to the next.\n"
            "4. Critical Gaps:\n"
            "   - Identify any missing elements that undermine the argument, such as insufficient evidence, "
            "     unaddressed counterarguments, or lack of context.\n"
            "   - Highlight areas where the argument is incomplete or where additional justification is needed.\n"
            "5. Suggestions for Improvement:\n"
            "   - Provide actionable recommendations to improve clarity, logical flow, and argument organization.\n"
            "   - Suggest revisions to strengthen weak or underdeveloped arguments.\n\n"
            "Your response should be structured in the following format:\n"
            "- Strengths: Highlight the well-written and logical aspects of the argument.\n"
            "- Issues: Identify specific problems in logical flow, clarity, or organization.\n"
            "- Suggestions: Provide detailed and actionable recommendations to address the issues.\n\n"
            "- Overall Assessment: Give an overall rating out of 10 of the argument's coherence and effectiveness.\n\n"
        )
        
        self.eval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "user", 
                    "This is the content of the research paper:\n\n {content}.\n\n"
                    "Please evaluate it and rate the overall quality of the paper out of 10."
                )
            ]
        )
        self.coherence_agent = self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_coherence(self, content_chunks):
        """
        Evaluates the coherence for each content chunk.

        Args:
            content_chunks (list): List of content chunks to evaluate.

        Returns:
            str: Combined evaluations of all content chunks.
        """
        evaluations = []
        for i, chunk in enumerate(content_chunks):
            print(f"Evaluating chunk {i + 1}/{len(content_chunks)}...")
            evaluation = self.coherence_agent.invoke({"content": chunk["content"]})
            evaluations.append(f"Evaluation for {chunk['heading']}:\n{evaluation}\n")
        return "\n".join(evaluations)


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
    pdf_path = "bad_paper.pdf"
    content_chunks = extract_and_chunk_paper(pdf_path)

    # Initialize Coherence Agent
    coherence_agent = CoherenceAgent(model="llama3-70b-8192", api_key="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")
    
    # Evaluate coherence for all chunks
    evaluation = coherence_agent.evaluate_coherence(content_chunks)
    
    # Write evaluation to file
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write(evaluation)
    
    print("Evaluation completed. Results saved to 'evaluation.txt'.")
