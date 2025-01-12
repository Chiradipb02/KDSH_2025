from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf

class EvidenceCheckerAgent:
    """
    Evaluates the logical consistency, clarity, and organization of arguments in the given content.

    Args:
        model (str): The model to be used for evaluation.
        api_key (str): The API key for accessing the model.
    """
    def __init__(self, model, api_key):
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.system_prompt = (
            "You are an expert evaluator specializing in evidence verification and critical analysis of arguments. "
            "Your task is to assess the credibility, sufficiency, and relevance of the evidence supporting the claims made in a text.\n\n"
            "When evaluating, follow these guidelines:\n"
            "1. Credibility:\n"
            "   - Evaluate the trustworthiness of the evidence provided.\n"
            "   - Identify if the evidence comes from credible, authoritative, and verifiable sources.\n"
            "   - Highlight any unsupported claims or reliance on questionable sources.\n"
            "2. Sufficiency:\n"
            "   - Assess whether the evidence provided is adequate to support the claims.\n"
            "   - Check if the claims are fully substantiated or if additional evidence is needed.\n"
            "3. Relevance:\n"
            "   - Determine whether the evidence is directly related to the claims made.\n"
            "   - Identify any tangential, irrelevant, or misleading information.\n"
            "4. Counterarguments:\n"
            "   - Check if counterarguments are addressed and whether evidence is provided to refute them.\n"
            "   - Highlight any biases or one-sided reasoning in the argument.\n"
            "5. Suggestions for Improvement:\n"
            "   - Provide actionable recommendations to strengthen the argument.\n"
            "   - Suggest additional evidence, sources, or clarifications needed to improve the credibility and sufficiency of the claims.\n\n"
            "Output Structure:\n"
            "- Strengths: Summarize what is well-supported and credible.\n"
            "- Issues: Identify specific problems with credibility, sufficiency, or relevance of the evidence.\n"
            "- Suggestions: Provide detailed and actionable recommendations to address the identified issues.\n"
            "- Overall Assessment: Provide an overall rating out of 10 on the quality of evidence supporting the claims."
        )
        self.eval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "This is the content to evaluate:\n\n{content}\n\nPlease verify the evidence supporting the claims and rate the quality of the paper out of 10 based on your review."),
            ]
        )
        self.evidence_checker = self.eval_prompt | self.llm | StrOutputParser()

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
            evaluation = self.evidence_checker.invoke({"content": chunk["content"]})
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
    pdf_path = "R004.pdf"
    content_chunks = extract_and_chunk_paper(pdf_path)

    
    evidence_agent = EvidenceCheckerAgent(model="llama3-70b-8192", api_key="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")
    
   
    evaluation = evidence_agent.evaluate_coherence(content_chunks)
    
    
    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write(evaluation)
    
    print("Evaluation completed. Results saved to 'evaluation.txt'.")
