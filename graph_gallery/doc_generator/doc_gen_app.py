import os
from dotenv import load_dotenv
import gradio as gr
from graph_gallery.doc_generator.doc_gen import DocGen

def doc_gen(input: str | None) -> str | None:
    """
    Generate a document on the topic prompted by the user.
    """
    doc_gen = DocGen()
    response = doc_gen.respond(input)
    return response

def main():
    load_dotenv(override=True)

    demo = gr.Interface(
        fn = doc_gen,
        inputs = gr.Textbox(
            lines = 2,
            placeholder="Enter your topic here...",
            label="Topic"
        ),
        outputs = gr.Textbox(
            lines = 30,
            label = "Generated Output",
            show_copy_button = True
        ),
        title = "Document Generator",
        description = "Enter a topic to generate a structured document",
        theme=gr.themes.Ocean(),
        flagging_mode="never"
    )

    demo.launch()    


if __name__ == "__main__":
    print("Running module directly (bypassing 'doc_gen' entry point)...")
    main()
