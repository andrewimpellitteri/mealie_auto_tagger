# Mealie Auto-Tagger


Automatically generate tags and categories for your [Mealie](https://mealie.io) recipe library using OpenAI LLMs.

This script uses AI to analyze your recipe ingredients and instructions, then suggests broad categories (like "Dinner", "Dessert") and specific tags (like "Chicken", "Gluten-Free", "Crispy").

## Features

- **AI-Powered Tagging**: Uses OpenAI's `gpt-4o-mini` for high-quality, singularized recipe tagging.
- **Robust API Integration**: Uses the verified "Full-Object Sync" pattern to avoid 400/422 validation errors common in the Mealie API.
- **Local Caching**: Saves results to `.mealie_cache.json` to prevent redundant API calls and save you money.
- **Progress Tracking**: Real-time progress bars for both recipe fetching and tagging.
- **Cost Efficient**: Optimized prompts and support for `gpt-4o-mini` to minimize usage costs.

## Prerequisites

- A running [Mealie](https://mealie.io) instance.
- A Mealie API Token (found in your User Profile > API Tokens).
- An OpenAI API Key.

## Installation & Usage

The recommended way to run this is using [uv](https://github.com/astral-sh/uv).

### 1. Configure Environment
Create a `.env` file in the project root:
```env
MEALIE_URL=http://your-mealie-url:9000
MEALIE_API_TOKEN=your_mealie_api_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Run
With `uv` installed, you don't even need to clone or install dependencies manually:

```bash
# Run everything
uv run mealie-auto-tagger.py

# Recommended: Only process recipes that don't have any tags yet
uv run mealie-auto-tagger.py --skip-tagged

# Testing: Limit to a specific number of recipes
uv run mealie-auto-tagger.py --limit=5
```

### Alternative: Standard Python
If you prefer standard `pip`:
```bash
pip install -r requirements.txt
python mealie-auto-tagger.py --skip-tagged
```

## How it Works

The script follows a robust multi-step process for each recipe:
1. **Fetch**: It retrieves the full recipe object from your Mealie instance.
2. **Analysis**: It sends the ingredients and instructions to OpenAI to generate logical categories and tags.
3. **Pre-creation**: It ensures all suggested tags/categories exist in Mealie *before* linking them.
4. **Sync**: It performs a full object update to the Mealie API to ensure all metadata (IDs, GroupIDs) is correctly preserved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
