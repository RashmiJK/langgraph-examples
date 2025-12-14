import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from graph_examples.logger import get_logger
from graph_examples.rag_search.chroma_interface import ChromaInterface
from graph_examples.rag_search.rag_search import RagSearch

load_dotenv(override=True)
logger = get_logger(__name__)


def ingest_documents(file_list: list[str], chunk_size: int) -> str:
    """
    Ingest documents and create embeddings.
    """
    if not file_list:
        return "Please upload a file first."

    ingest_chroma = ChromaInterface.get_instance()
    ingestion_status = []

    for file in file_list:
        status = ingest_chroma.ingest(file, chunk_size)
        ingestion_status.append(status)

    return "\n".join(ingestion_status)


def describe_ingested_files() -> str:
    """
    List of ingested files.
    """
    describe_chroma = ChromaInterface.get_instance()
    return describe_chroma.describe_ingested_content()


def backend_search_pipeline(
    query: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Mock function to simulate search.
    """
    logger.debug("Query: %s", query)

    # validate query
    if not query or not query.strip():
        return "Please enter a search query.", pd.DataFrame(), pd.DataFrame(), "", ""

    rag_search = RagSearch()
    search_results, reranked_results, baseline_answer, reranked_answer = (
        rag_search.respond(query)
    )
    logger.debug("Search results:\n %s %s", search_results, type(search_results))
    logger.info("Number of search results: %d", len(search_results.results))

    search_result_df = pd.DataFrame(
        {"Score": r.score, "Chunk Text": r.document, "Source": r.source}
        for r in search_results.results
    )
    # search_result_df.sort_values(by="Score", inplace=True).reset_index(drop=True)
    logger.debug("Search results df:\n %s", search_result_df)

    logger.debug("Reranker results:\n %s %s", reranked_results, type(reranked_results))
    logger.info("Number of reranker results: %d", len(reranked_results))
    reranked_result_df = pd.DataFrame(
        {
            "Reranker Score": round(float(r["score"]), 2),
            "Chunk Text": r["text"],
            "Source": r["meta"]["source"],
        }
        for r in reranked_results
    )
    # search_result_df.sort_values(by="Score", inplace=True).reset_index(drop=True)
    logger.debug("Reranker results df:\n %s", reranked_result_df)

    return query, search_result_df, reranked_result_df, baseline_answer, reranked_answer


def main() -> None:
    """
    Main function to run the app.
    """
    title = os.path.splitext(os.path.basename(__file__))[0]

    with gr.Blocks(title=title) as demo:
        # Header
        gr.Markdown(
            """
            # 🧠 Document Search & Reranking Analysis
            Upload a document, configure chunking, and compare initial retrieval vs. reranked results.
            """
        )
        with gr.Row():
            # Column 1: File upload & Indexing
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 1. Data Setup")

                # File Upload
                file_input = gr.File(
                    label="Upload Document (PDF, TXT, MD)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                )

                gr.HTML("<div style='height: 20px;'></div>")

                # Chunk Size Selection
                chunk_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=512,
                    step=128,
                    label="Choose Chunk Size",
                    info="Larger chunks capture more context, smaller chunks are more specific.",
                )

                # Indexing Button
                index_btn = gr.Button("🛠️ Ingest Documents", variant="secondary")

                # Indexing Status Feedback
                index_status = gr.Textbox(
                    label="Ingestion Status",
                    placeholder="Waiting for document...",
                    interactive=False,
                    lines=2,
                )

                gr.HTML("<div style='height: 40px;'></div>")

                # Describe Ingested Content
                describe_btn = gr.Button(
                    "Describe Ingested Content", variant="secondary"
                )

                describe_status = gr.Textbox(
                    label="List of ingested files",
                    interactive=False,
                    lines=2,
                )

            # Column 2: Search & Reranking
            with gr.Column(scale=2):
                gr.Markdown("### 2. Query & Analysis")

                # User Query
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your question here...",
                        scale=4,
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary", scale=1)

                # Results Comparison Section
                with gr.Row():
                    # Initial Retrieval Results
                    with gr.Column():
                        gr.HTML(
                            "<h4 style='color: gray; margin-bottom: 10px;'>Stage 1: Initial Retrieval (Top 6)</h4>"
                        )
                        # Using Dataframe for clean display of scores and text
                        initial_results_output = gr.Dataframe(
                            headers=["Score", "Chunk Text", "Source"],
                            datatype=["number", "str", "str"],
                            row_count="dynamic",
                            column_count=3,
                            column_widths=["80px", "600px", "200px"],
                            max_height=None,
                            interactive=False,
                            wrap=True,
                            show_row_numbers=True,
                            # elem_classes=["bordered-table"],
                        )
                        answer_baseline_output = gr.Textbox(
                            label="Answer (Uses Top 2 Search Results)",
                            placeholder="Answer will be displayed here...",
                            interactive=False,
                            lines=4,
                        )

                with gr.Row():
                    # Reranker Output
                    with gr.Column():
                        gr.HTML(
                            "<h4 style='color: #2563eb; margin-bottom: 10px;'>Stage 2: After Reranking (Top 6)</h4>"
                        )
                        reranked_results_output = gr.Dataframe(
                            headers=["ReRanker Score", "Chunk Text", "Source"],
                            datatype=["number", "str", "str"],
                            row_count="dynamic",
                            column_count=3,
                            column_widths=["80px", "600px", "200px"],
                            max_height=None,
                            interactive=False,
                            wrap=True,
                            show_row_numbers=True,
                            # elem_classes=["bordered-table"],
                        )
                        answer_reranked_output = gr.Textbox(
                            label="Answer (Uses Top 2 RerankedResults)",
                            placeholder="Answer will be displayed here...",
                            interactive=False,
                            lines=4,
                        )

        # Wire the Index Button
        index_btn.click(
            fn=ingest_documents,
            inputs=[file_input, chunk_slider],
            outputs=[index_status],
            show_progress="full",
        )

        # Wire the Search Button & Enter key submission
        search_triggers = [search_btn.click, query_input.submit]
        for trigger in search_triggers:
            trigger(
                fn=backend_search_pipeline,
                inputs=[query_input],
                outputs=[
                    query_input,
                    initial_results_output,
                    reranked_results_output,
                    answer_baseline_output,
                    answer_reranked_output,
                ],
                show_progress="full",
            )

        # Wire the Describe button
        describe_btn.click(
            fn=describe_ingested_files,
            inputs=None,
            outputs=[describe_status],
            show_progress="full",
        )

        demo.launch(
            theme=gr.themes.Soft(
                primary_hue=gr.themes.colors.sky,
                secondary_hue=gr.themes.colors.amber,
                neutral_hue=gr.themes.colors.slate,
            ).set(
                body_background_fill="#0c1222",
                body_background_fill_dark="#0c1222",
                block_background_fill="#162032",
                block_background_fill_dark="#162032",
                button_primary_background_fill="linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)",
                button_primary_background_fill_hover="linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%)",
                button_secondary_background_fill="#1e3a5f",
                button_secondary_background_fill_hover="#2a4a73",
                border_color_primary="#1e3a5f",
                input_background_fill="#162032",
                block_label_background_fill="*secondary_600",
                block_label_text_color="white",
            ),
            css_paths=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
            ],
        )


if __name__ == "__main__":
    main()
