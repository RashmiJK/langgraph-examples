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
```
The examples are currently configured to use GitHub Models via AzureAIChatCompletionsModel.

If you wish to use a different provider (e.g., OpenAI, Azure OpenAI, or Anthropic), code changes are required.

Example:
1. Open graph_examples/doc_generator/doc_gen.py
2. Update the DocGen class initialization to use your preferred LangChain chat model.
3. Update the model parameters to match your provider's available models.

## Running the Examples
<table border="1">
    <tr>
        <!-- This cell spans 2 columns -->
        <td colspan="2" align="center"><strong><h2>LangGraph Examples</h2></strong></td>
    </tr>
    <tr>
        <td> Document Generator</td>
        <td> Next Example </td>
    </tr>
    <tr>
        <td>
            <a href="src/graph_examples/doc_generator/doc_gen_graph.png"> <img src="src/graph_examples/doc_generator/doc_gen_graph.png" width="200" height="200"></a>   
            <a href="src/graph_examples/doc_generator/doc_gen_ui.png"> <img src="src/graph_examples/doc_generator/doc_gen_ui.png" width="200" height="200"></a>
        </td>
        <td>            
        </td>
    </tr>
    <tr>
        <td>From the repository root folder, execute: <code>uv run doc_gen</code></td>
        <td></td>
    </tr>
</table>