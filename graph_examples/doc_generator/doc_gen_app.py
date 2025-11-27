import os
from dotenv import load_dotenv
import gradio as gr
from graph_examples.doc_generator.doc_gen import DocGen
from fpdf import FPDF
import tempfile

async def doc_gen(input: str | None) -> tuple[str | None, str | None]:
    """
    Generate a document on the topic prompted by the user (async).
    """
    doc_gen = DocGen()
    # Await the async respond method
    response, eval_summary = await doc_gen.respond(input)
    return response, eval_summary

def save_as_pdf(output_text: str):
    """
    Save the output text as a PDF file.
    """
    # Validate output-text
    if not output_text or output_text.strip() == "":
        return None
    
    # Create a PDF object
    pdf = FPDF()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, "fonts", "DejaVuSans.ttf")
    
    pdf.add_page()

    try:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size = 12)
    except RuntimeError:
        print(f"Warning: Could not load font at {font_path}. Fallback to standard Arial.")
        pdf.set_font("Arial", size = 12)    

    # Add content to PDF
    # Handle line breaks and ensure text fits within page width
    for line in output_text.split("\n"):
        if line.strip(): # skip empty lines
            pdf.multi_cell(0, 6, txt=line, align="L")
        else:
            pdf.ln(3)        

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix='document_')
    pdf.output(temp_file.name)  
    
    return temp_file.name
    

def main():
    load_dotenv(override=True)

    # Use gr.blocks for custom layout
    with gr.Blocks(
            theme=gr.themes.Ocean(),
            css="""
                #eval-summary {
                    color: #06b6d4;
                }
            """
        ) as demo:
        gr.Markdown(
                "<h1 style='text-align: center; color: #06b6d4'> Document Generator</h1>"
            )
        gr.Markdown("<p style='color: #06b6d4'>Enter a topic to generate a structured document</p>")

        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    lines = 2,
                    placeholder="Enter your topic here...",
                    label="Topic"
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
            
            # Right Column - Output
            with gr.Column(scale=1):
                output_box = gr.Textbox(
                    lines = 30,
                    label = "Generated Output",
                    show_copy_button = True
                )

                eval_summary_md = gr.Markdown(elem_id="eval-summary")

                save_pdf_btn = gr.Button("💾 Download as PDF", variant="primary", visible=True)
                pdf_file = gr.File(visible=False)

        # Event handlers
        submit_btn.click(
            fn = doc_gen, # Gradio handles Async function execution
            inputs = input_box,
            outputs = [output_box, eval_summary_md]
        )

        clear_btn.click(
            fn = lambda: ("", ""),
            inputs=None,
            outputs = (input_box, output_box, eval_summary_md)
        )

        # After click hide button and show file
        save_pdf_btn.click(
            fn = save_as_pdf,
            inputs=output_box,
            outputs=pdf_file
        ).then(
            fn=lambda: (gr.Button(visible=False), gr.File(visible=True)),
            outputs=[save_pdf_btn, pdf_file]
        )
        
        # Allow enter key to submit
        input_box.submit(
            fn = doc_gen,
            inputs = input_box,
            outputs = [output_box, eval_summary_md]
        )

        demo.launch()

if __name__ == "__main__":
    main()
