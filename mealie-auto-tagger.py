#!/usr/bin/env python3
# /// script
# dependencies = [
#   "openai>=1.55.0",
#   "requests>=2.31.0",
#   "tqdm>=4.66.0",
#   "python-dotenv>=1.0.0",
# ]
# ///
"""
Mealie Auto-Tagger
Automatically generates tags and categories for Mealie recipes using OpenAI GPT-5 nano.
"""

import os
import sys
import json
import requests
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import time
import re
from dotenv import load_dotenv


class MealieAutoTagger:
    def __init__(self, mealie_url: str, mealie_token: str, openai_api_key: str, voting_rounds: int = 1, cache_file: str = ".mealie_cache.json"):
        """Initialize the auto-tagger with API credentials and local cache."""
        self.mealie_url = mealie_url.rstrip('/')
        self.mealie_token = mealie_token
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.voting_rounds = voting_rounds
        self.headers = {
            'Authorization': f'Bearer {self.mealie_token}',
            'Content-Type': 'application/json'
        }
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.organizer_cache = {'tags': None, 'categories': None}

    def _load_cache(self) -> Dict:
        """Load the local results cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save the current cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _get_recipe_hash(self, recipe_text: str) -> str:
        """Generate a stable hash for the recipe content."""
        import hashlib
        return hashlib.sha256(recipe_text.encode('utf-8')).hexdigest()

    def get_existing_tags(self) -> List[str]:
        """Fetch all existing tags from Mealie to provide as context to the LLM."""
        try:
            response = requests.get(
                f'{self.mealie_url}/api/organizers/tags',
                headers=self.headers
            )
            response.raise_for_status()
            tags_data = response.json()
            
            # Extract tags from both list and paginated formats
            if isinstance(tags_data, list):
                tags = tags_data
            elif isinstance(tags_data, dict) and 'items' in tags_data:
                tags = tags_data['items']
            else:
                return []
                
            return [tag.get('name', tag.get('slug', '')) for tag in tags]
        except Exception as e:
            print(f"Warning: Could not fetch existing tags: {e}")
            return []

    def _get_organizer_items(self, item_type: str) -> List[Dict]:
        """Fetch all items (tags or categories) with full metadata (cached in memory)."""
        if self.organizer_cache.get(item_type) is not None:
            return self.organizer_cache[item_type]

        try:
            response = requests.get(
                f'{self.mealie_url}/api/organizers/{item_type}',
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'items' in data:
                items = data['items']
            
            self.organizer_cache[item_type] = items
            return items
        except Exception as e:
            print(f"Error fetching {item_type}: {e}")
            return []

    def _ensure_organizer_item(self, name: str, item_type: str) -> Optional[Dict]:
        """
        Ensures a tag or category exists in Mealie. If it doesn't, it creates it.
        Returns the full object for the item.
        item_type can be 'tags' or 'categories'.
        """
        # 1. Fetch existing items
        items = self._get_organizer_items(item_type)
        
        # 2. Check if it already exists (case-insensitive)
        item_slug = self._slugify(name)
        for item in items:
            if item.get('slug') == item_slug or item.get('name', '').lower() == name.lower():
                return item
        
        # 3. Create it if it doesn't exist
        display_type = "category" if item_type == "categories" else "tag"
        print(f"  Creating new {display_type}: '{name}'")
        try:
            response = requests.post(
                f'{self.mealie_url}/api/organizers/{item_type}',
                headers=self.headers,
                json={'name': name}
            )
            response.raise_for_status()
            new_item = response.json()
            
            # Update memory cache so subsequent calls don't need a re-fetch
            if self.organizer_cache.get(item_type) is not None:
                self.organizer_cache[item_type].append(new_item)
                
            return new_item
        except Exception as e:
            print(f"Error creating {display_type} '{name}': {e}")
            return None

    def get_recipes(self, page: int = 1, per_page: int = 50) -> Dict:
        """Fetch recipes from Mealie with pagination."""
        try:
            response = requests.get(
                f'{self.mealie_url}/api/recipes',
                headers=self.headers,
                params={'page': page, 'perPage': per_page}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching recipes: {e}")
            return {'items': [], 'total': 0}

    def get_recipe_details(self, recipe_slug: str) -> Optional[Dict]:
        """Fetch full recipe details including ingredients and instructions."""
        try:
            response = requests.get(
                f'{self.mealie_url}/api/recipes/{recipe_slug}',
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching recipe {recipe_slug}: {e}")
            return None

    def _validate_tag_output(self, result: Dict) -> bool:
        """Validate that LLM output follows the expected format and rules."""
        if not isinstance(result, dict):
            return False

        if 'categories' not in result or 'tags' not in result:
            return False

        if not isinstance(result['categories'], list) or not isinstance(result['tags'], list):
            return False

        # Check for invalid patterns in tags
        invalid_patterns = [
            'thickened with', 'reduced', 'not ', 'non-',
            'without', 'free of', 'doesn\'t', 'don\'t'
        ]

        all_items = result['categories'] + result['tags']
        for item in all_items:
            if not isinstance(item, str):
                return False

            # Check for invalid patterns
            item_lower = item.lower()
            for pattern in invalid_patterns:
                if pattern in item_lower:
                    return False
        return True

    def _combine_votes(self, results: List[Dict]) -> Dict:
        """Combine multiple LLM outputs using voting."""
        from collections import Counter

        category_votes = Counter()
        tag_votes = Counter()

        for result in results:
            if not self._validate_tag_output(result):
                continue

            for cat in result['categories']:
                category_votes[cat.lower()] += 1

            for tag in result['tags']:
                tag_votes[tag.lower()] += 1

        # Require at least 50% votes (majority)
        threshold = len(results) / 2

        winning_categories = [cat for cat, votes in category_votes.items() if votes > threshold]
        winning_tags = [tag for tag, votes in tag_votes.items() if votes > threshold]

        return {
            'categories': winning_categories,
            'tags': winning_tags
        }

    def generate_tags(self, recipe: Dict, existing_tags: List[str]) -> Optional[Dict]:
        """Use OpenAI to generate tags and categories for a recipe (with local caching)."""
        # Prepare recipe text for the LLM
        recipe_text = self._format_recipe_for_llm(recipe)
        
        # Check cache first
        recipe_hash = self._get_recipe_hash(recipe_text)
        if recipe_hash in self.cache:
            return self.cache[recipe_hash]

        # Build the prompt with existing tags
        existing_tags_str = json.dumps(existing_tags, indent=2)

        prompt = f"""Generate "categories" (broad) and "tags" (specific) for this recipe as JSON.
Existing tags to reuse if applicable: {existing_tags_str}

Rules:
- Tags in SINGULAR form.
- No preparation details or negative tags.
- Output JSON format: {{"categories": [], "tags": []}}

Recipe:
{recipe_text}"""

        # Generate multiple responses if using voting
        results = []
        max_retries = 3

        for round_num in range(self.voting_rounds):
            retries = 0
            while retries < max_retries:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=[
                            {"role": "system", "content": "You are a recipe categorizer. Return only JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )

                    result = json.loads(response.choices[0].message.content)

                    # Validate the output
                    if self._validate_tag_output(result):
                        results.append(result)
                        break
                    else:
                        retries += 1
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        tqdm.write(f"  Error generating tags: {e}")
                        break

        if not results:
            return None

        # Combine results or pick first
        final_result = self._combine_votes(results) if self.voting_rounds > 1 else results[0]
        
        # Save to cache
        self.cache[recipe_hash] = final_result
        self._save_cache()
        
        return final_result

    def _format_recipe_for_llm(self, recipe: Dict) -> str:
        """Format recipe data into readable text for the LLM."""
        parts = []

        # Title
        if recipe.get('name'):
            parts.append(f"Title: {recipe['name']}")

        # Description
        if recipe.get('description'):
            parts.append(f"\nDescription: {recipe['description']}")

        # Ingredients
        if recipe.get('recipeIngredient') or recipe.get('ingredients'):
            parts.append("\nIngredients:")
            ingredients = recipe.get('recipeIngredient', recipe.get('ingredients', []))
            for ing in ingredients:
                if isinstance(ing, str):
                    parts.append(f"- {ing}")
                elif isinstance(ing, dict):
                    note = ing.get('note', ing.get('display', ''))
                    if note:
                        parts.append(f"- {note}")

        # Instructions
        if recipe.get('recipeInstructions') or recipe.get('instructions'):
            parts.append("\nInstructions:")
            instructions = recipe.get('recipeInstructions', recipe.get('instructions', []))
            for idx, inst in enumerate(instructions, 1):
                if isinstance(inst, str):
                    parts.append(f"{idx}. {inst}")
                elif isinstance(inst, dict):
                    text = inst.get('text', inst.get('title', ''))
                    if text:
                        parts.append(f"{idx}. {text}")

        return "\n".join(parts)

    def _slugify(self, text: str) -> str:
        """Convert text to a URL-safe slug."""
        # Lowercase and replace spaces/special chars with hyphens
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars except hyphens
        slug = re.sub(r'[\s_]+', '-', slug)   # Replace spaces/underscores with hyphens
        slug = re.sub(r'-+', '-', slug)       # Collapse multiple hyphens
        return slug.strip('-')

    def update_recipe_tags(self, recipe_slug: str, tags: List[str], categories: List[str]) -> bool:
        """Update a recipe with new tags and categories using the verified full-object pattern."""
        try:
            # 1. Ensure all tags and categories exist and get their full metadata
            tag_objects = []
            for tag_name in tags:
                tag_obj = self._ensure_organizer_item(tag_name, 'tags')
                if tag_obj:
                    tag_objects.append(tag_obj)
            
            category_objects = []
            for cat_name in categories:
                cat_obj = self._ensure_organizer_item(cat_name, 'categories')
                if cat_obj:
                    category_objects.append(cat_obj)

            # 2. Fetch the ABSOLUTELY CURRENT full recipe object by slug
            response = requests.get(
                f'{self.mealie_url}/api/recipes/{recipe_slug}',
                headers=self.headers
            )
            response.raise_for_status()
            recipe_payload = response.json()

            # 3. Update tags and categories
            recipe_payload['tags'] = tag_objects
            recipe_payload['recipeCategory'] = category_objects

            # 4. Save the ENTIRE object back to the slug-based endpoint
            response = requests.put(
                f'{self.mealie_url}/api/recipes/{recipe_slug}',
                headers=self.headers,
                json=recipe_payload
            )
            
            if response.status_code != 200:
                print(f"Error updating recipe {recipe_slug} (Status {response.status_code}): {response.text[:500]}")
                return False

            return True

        except Exception as e:
            print(f"Unexpected error updating recipe {recipe_slug}: {e}")
            return False

    def process_all_recipes(self, skip_tagged: bool = False, limit: Optional[int] = None):
        """Process all recipes and generate tags."""
        print("Fetching existing tags from Mealie...")
        existing_tags = self.get_existing_tags()
        print(f"Found {len(existing_tags)} existing tags")

        print("\nFetching recipes...")
        all_recipes = []
        page = 1
        pbar = None

        while True:
            result = self.get_recipes(page=page, per_page=50)
            items = result.get('items', [])
            if not items:
                break
            
            if pbar is None:
                total = result.get('total', len(items))
                pbar = tqdm(total=total, desc="Fetching recipes", leave=False)
            
            all_recipes.extend(items)
            pbar.update(len(items))

            if len(all_recipes) >= pbar.total:
                break
            page += 1
            time.sleep(0.5)  # Be nice to the API
        
        if pbar:
            pbar.close()

        print(f"Found {len(all_recipes)} total recipes")

        # Filter recipes if needed
        recipes_to_process = []
        for recipe in all_recipes:
            if skip_tagged:
                # Skip recipes that already have tags
                if recipe.get('tags') and len(recipe['tags']) > 0:
                    continue
            recipes_to_process.append(recipe)

        if limit:
            recipes_to_process = recipes_to_process[:limit]

        print(f"\nProcessing {len(recipes_to_process)} recipes...\n")

        # Process each recipe with progress bar
        successful = 0
        failed = 0

        for recipe in tqdm(recipes_to_process, desc="Tagging recipes"):
            try:
                recipe_slug = recipe.get('slug') or recipe.get('id')
                recipe_name = recipe.get('name', recipe_slug)

                # Get full recipe details
                full_recipe = self.get_recipe_details(recipe_slug)
                if not full_recipe:
                    failed += 1
                    continue

                # Generate tags using OpenAI
                result = self.generate_tags(full_recipe, existing_tags)
                if not result:
                    failed += 1
                    continue

                tags = result.get('tags', [])
                categories = result.get('categories', [])

                # Update the recipe
                if self.update_recipe_tags(recipe_slug, tags, categories):
                    successful += 1
                    tqdm.write(f"✓ {recipe_name}: {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}")
                else:
                    failed += 1

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                tqdm.write(f"✗ Error processing {recipe.get('name', 'unknown')}: {e}")
                failed += 1

        print(f"\n{'='*60}")
        print(f"Complete! Successfully tagged {successful} {'recipe' if successful == 1 else 'recipes'}")
        if failed > 0:
            print(f"Failed to tag {failed} {'recipe' if failed == 1 else 'recipes'}")
        print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    # Load configuration from environment variables
    load_dotenv()
    mealie_url = os.getenv('MEALIE_URL')
    mealie_token = os.getenv('MEALIE_API_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Validate configuration
    if not mealie_url:
        print("Error: MEALIE_URL environment variable not set")
        sys.exit(1)

    if not mealie_token:
        print("Error: MEALIE_API_TOKEN environment variable not set")
        sys.exit(1)

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Parse command line arguments
    skip_tagged = '--skip-tagged' in sys.argv
    limit = None
    voting_rounds = 1

    for arg in sys.argv[1:]:
        if arg.startswith('--limit='):
            try:
                limit = int(arg.split('=')[1])
            except ValueError:
                print(f"Invalid limit value: {arg}")
                sys.exit(1)
        elif arg.startswith('--voting-rounds='):
            try:
                voting_rounds = int(arg.split('=')[1])
            except ValueError:
                print(f"Invalid voting-rounds value: {arg}")
                sys.exit(1)

    # Print configuration
    print("="*60)
    print("Mealie Auto-Tagger")
    print("="*60)
    print(f"Mealie URL: {mealie_url}")
    print(f"OpenAI Model: gpt-5-nano")
    print(f"Skip already tagged: {skip_tagged}")
    if limit:
        print(f"Limit: {limit} recipes")
    print("="*60)
    print()

    # Create tagger and process recipes
    tagger = MealieAutoTagger(mealie_url, mealie_token, openai_api_key, voting_rounds=voting_rounds)
    tagger.process_all_recipes(skip_tagged=skip_tagged, limit=limit)


if __name__ == '__main__':
    main()
