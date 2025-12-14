# LangGraph Examples


## Environment Setup

Use [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/) to run the examples.

```bash
# Install uv on MacOS
brew install uv

# Install uv on Windows
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Clone the Git repository
git clone <this_repository_url>
```

```bash
# Sync the environment 
uv sync
```

## Configuration

Create a `.env` file in the root directory and add your GitHub inference credentials:

```bash
GITHUB_INFERENCE_ENDPOINT="<github_inference_endpoint>"
GITHUB_TOKEN="<github_token>"
LOG_LEVEL="<log_level>"
OPIK_API_KEY="<opik_api_key>" # Optional for observability
```
The examples are currently configured to use GitHub Models via AzureAIChatCompletionsModel.

If you wish to use a different provider (e.g., OpenAI, Azure OpenAI, or Anthropic), code changes are required.

Example:
1. Open src/graph_examples/doc_generator/doc_gen.py
2. Update the DocGen class initialization to use your preferred LangChain chat model.
3. Update the model parameters to match your provider's available models.

## Running the Examples
<table border="1">
    <tr>
        <!-- This cell spans 2 columns -->
        <td colspan="2" align="center"><strong><h2>LangGraph Examples</h2></strong></td>
    </tr>
    <tr>
        <td> <b>Document Generator</b></td>
        <td> <b>Search and Reranking Analysis</b></td>
    </tr>
    <tr>
        <td>
            <a href="src/graph_examples/doc_generator/doc_gen_graph.png"> <img src="src/graph_examples/doc_generator/doc_gen_graph.png" width="175" height="175"></a>   
            <a href="src/graph_examples/doc_generator/doc_gen_ui.png"> <img src="src/graph_examples/doc_generator/doc_gen_ui.png" width="175" height="175"></a>
        </td>
        <td>    
            <a href="src/graph_examples/rag_search/rag_search_graph.png"> <img src="src/graph_examples/rag_search/rag_search_graph.png" width="175" height="175"></a>   
            <a href="src/graph_examples/rag_search/rag_search_ui.png"> <img src="src/graph_examples/rag_search/rag_search_ui.png" width="175" height="175"></a>
        </td>
    </tr>
    <tr>
        <td><i>From the repository root folder, execute: <code>uv run doc_gen</code></i></td>
        <td><i>From the repository root folder, execute: <code>uv run rag_search</code></i></td>
    </tr>
    <tr>
    <td colspan="2" style="height: 10px"></td>
    </tr>
    <tr>
        <td> <b>Next flow </b></td>
        <td> <b>Next flow </b></td>
    </tr>
</table>

## 💡 Heads Up
This repo grows as I learn. Mistakes happen, updates will follow, and perfection is not the goal — exploration is. Use it, break it, make it better!