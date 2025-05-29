import os
import random
import time
import requests
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import base64
import logging
from datetime import datetime
import unicodedata
import platform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blog_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# CONFIGURATION
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CATEGORIES = ["technology", "health", "sports", "business", "entertainment"]
NUM_SOURCE_ARTICLES_TO_AGGREGATE = 5 
LANGUAGE = 'en'
BRANDING_LOGO_PATH = os.getenv('BRANDING_LOGO_PATH', None)
IMAGE_OUTPUT_FOLDER = "transformed_images"
BLOG_OUTPUT_FOLDER = "blog_drafts"

# LLM Retry Configuration
LLM_MAX_RETRIES = 5
LLM_INITIAL_RETRY_DELAY_SECONDS = 5

# Enhanced font configuration with fallbacks
FONT_PATHS = {
    'mac': [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFProText-Regular.ttf"
    ],
    'windows': [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    ],
    'linux': [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    ]
}

def find_system_font():
    """Find the best available font for the current system"""
    system = platform.system().lower()
    
    if 'darwin' in system:
        font_list = FONT_PATHS['mac']
    elif 'windows' in system:
        font_list = FONT_PATHS['windows']
    else:
        font_list = FONT_PATHS['linux']
    
    for font_path in font_list:
        if os.path.exists(font_path):
            logger.info(f"Using font: {font_path}")
            return font_path
    
    logger.warning("No suitable system fonts found, using PIL default.")
    return None

DEFAULT_FONT_PATH = find_system_font()

# Create necessary directories
for folder in [IMAGE_OUTPUT_FOLDER, BLOG_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(folder, cat), exist_ok=True)

# Setup Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    RESEARCH_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20') 
    CONTENT_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
else:
    logger.error("GEMINI_API_KEY not set. Gemini functions will not work.")

def validate_environment():
    """Validate that all required environment variables and dependencies are set"""
    errors = []
    
    if not GNEWS_API_KEY:
        errors.append("GNEWS_API_KEY not found in environment variables.")
    
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY not found. Gemini functions will be skipped.")
    
    try:
        import PIL
        import google.generativeai
        import requests
    except ImportError as e:
        errors.append(f"Missing required package: {e}.")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    logger.info("Environment validation passed.")
    return True

def sanitize_filename(filename):
    """Create a safe filename from any string"""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in filename).strip()
    safe_title = re.sub(r'[_ -]+', '_', safe_title).lower()
    return safe_title[:100]

def fetch_gnews_articles(category, max_articles_to_fetch=10, max_retries=3):
    """Fetches articles from GNews API with retry logic"""
    url = 'https://gnews.io/api/v4/top-headlines'
    params = {
        'category': category,
        'lang': LANGUAGE,
        'token': GNEWS_API_KEY,
        'max': max_articles_to_fetch
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching up to {max_articles_to_fetch} articles for {category} (attempt {attempt + 1})...")
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            
            data = resp.json()
            articles = data.get('articles', [])
            
            if not articles:
                logger.warning(f"No articles found for category {category}.")
                return []
            
            unique_articles = {article['url']: article for article in articles}.values()
            selected_articles = list(unique_articles)[:max_articles_to_fetch]
            logger.info(f"Successfully fetched {len(selected_articles)} articles for {category}.")
            return selected_articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching articles (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []

def aggregate_articles(articles_list, category):
    """
    Aggregates data from multiple articles to create a consolidated view
    for a single, unique blog post.
    """
    if not articles_list:
        logger.warning(f"No articles provided for aggregation in {category}.")
        return None

    consolidated_content = []
    consolidated_descriptions = []
    titles = []
    image_url = None
    competitor_domains = set()
    primary_source_url_for_disclaimer = None

    sorted_articles = sorted(articles_list, key=lambda x: len(x.get('content', '')), reverse=True)

    for i, article in enumerate(sorted_articles):
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        source_url = article.get('url', '')
        article_image = article.get('image', '')
        source_domain = article.get('source', {}).get('url', '').replace('https://', '').replace('http://', '').split('/')[0]

        if title: titles.append(title)
        if description: consolidated_descriptions.append(description)
        
        if content and content.strip() != '[Removed]' and len(content.strip()) > 50:
            consolidated_content.append(f"### Source: {title}\n\n{content}")
            
            if not image_url and article_image and len(content) > 100: 
                image_url = article_image
                primary_source_url_for_disclaimer = source_url
        
        if source_domain:
            competitor_domains.add(source_domain)
        
    if not image_url:
        for article in articles_list:
            if article.get('image'):
                image_url = article['image']
                primary_source_url_for_disclaimer = article['url']
                break

    if not image_url:
        logger.warning(f"No valid image URL found for {category}.")

    consolidated_topic = titles[0] if titles else f"Recent Developments in {category.capitalize()}"
    if len(titles) > 1:
        combined_titles_string = " ".join(titles[:min(3, len(titles))])
        consolidated_topic = f"Comprehensive Look: {combined_titles_string}"
        if len(consolidated_topic) > 150:
            consolidated_topic = consolidated_topic[:150] + "..."
        consolidated_topic = consolidated_topic.strip()

    if not consolidated_descriptions:
        consolidated_descriptions.append(f"A deep dive into recent developments in {category}.")

    return {
        "consolidated_topic": consolidated_topic,
        "combined_content": "\n\n---\n\n".join(consolidated_content) if consolidated_content else "No substantial content found.",
        "combined_description": " ".join(consolidated_descriptions)[:300].strip(),
        "image_url": image_url,
        "competitors": list(competitor_domains),
        "primary_source_url": primary_source_url_for_disclaimer if primary_source_url_for_disclaimer else articles_list[0]['url'] if articles_list else 'https://news.example.com/source-unavailable'
    }

def enhance_image_quality(img):
    """Apply advanced image enhancement techniques."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)
    
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    
    return img

def create_text_with_shadow(draw, position, text, font, text_color, shadow_color, shadow_offset):
    """Draw text with shadow for better visibility."""
    x, y = position
    shadow_x, shadow_y = shadow_offset
    
    draw.text((x + shadow_x, y + shadow_y), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=text_color)

def find_content_bbox_and_trim(img, tolerance=20, border_colors_to_trim=((0,0,0), (255,255,255))):
    """
    Attempts to find the bounding box of non-border content pixels and trims the image.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size
    pixels = img.load()

    def is_similar(pixel1, pixel2, tol):
        return all(abs(c1 - c2) <= tol for c1, c2 in zip(pixel1, pixel2))

    def is_border_pixel_group(pixel):
        return any(is_similar(pixel, bc, tolerance) for bc in border_colors_to_trim)
    
    top = 0
    for y in range(height):
        if not all(is_border_pixel_group(pixels[x, y]) for x in range(width)):
            top = y
            break
    
    bottom = height
    for y in range(height - 1, top, -1):
        if not all(is_border_pixel_group(pixels[x, y]) for x in range(width)):
            bottom = y + 1
            break

    left = 0
    for x in range(width):
        if not all(is_border_pixel_group(pixels[x, y]) for y in range(height)):
            left = x
            break

    right = width
    for x in range(width - 1, left, -1):
        if not all(is_border_pixel_group(pixels[x, y]) for y in range(height)):
            right = x + 1
            break
    
    if (left, top, right, bottom) != (0, 0, width, height):
        trimmed_width = right - left
        trimmed_height = bottom - top
        min_content_ratio = 0.75
        if trimmed_width > (width * min_content_ratio) and \
           trimmed_height > (height * min_content_ratio):
            logger.info(f"Automatically trimmed detected uniform borders.")
            return img.crop((left, top, right, bottom))
        else:
            logger.debug("Trimming borders would remove too much content.")
    
    logger.debug("No significant uniform color borders detected.")
    return img

def transform_image(image_url, title_text, category_text, output_category_folder, safe_filename):
    """
    Downloads, processes, and adds branding/text to an image.
    Returns (relative_file_path, base64_data_uri) or (None, None) on failure.
    """
    if not image_url:
        logger.info("No image URL provided. Skipping image processing.")
        return None, None

    output_full_path = None
    base64_data_uri = None

    try:
        logger.info(f"Processing image from URL: {image_url[:70]}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=20, stream=True, headers=headers)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        if img.mode in ('RGBA', 'LA', 'P'):
            alpha = img.split()[-1] if img.mode in ('RGBA', 'LA') else None
            background = Image.new('RGB', img.size, (255, 255, 255))
            if alpha:
                background.paste(img, mask=alpha)
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img = find_content_bbox_and_trim(img)

        target_content_width = 1200
        target_content_height = 675
        target_aspect = target_content_width / target_content_height

        original_width, original_height = img.size
        original_aspect = original_width / original_height

        if original_aspect > target_aspect:
            resize_height = target_content_height
            resize_width = int(target_content_height * original_aspect)
        else:
            resize_width = target_content_width
            resize_height = int(target_content_width / original_aspect)

        img = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

        left_crop = (resize_width - target_content_width) // 2
        top_crop = (resize_height - target_content_height) // 2
        img = img.crop((left_crop, top_crop, left_crop + target_content_width, top_crop + target_content_height))

        img = enhance_image_quality(img)
        img = img.convert('RGBA')
        
        extended_area_height = int(target_content_height * 0.25)
        final_canvas_height = target_content_height + extended_area_height
        final_canvas_width = target_content_width

        new_combined_img = Image.new('RGBA', (final_canvas_width, final_canvas_height), (0, 0, 0, 255))
        new_combined_img.paste(img, (0, 0))

        strip_from_original_height = int(target_content_height * 0.05)
        if strip_from_original_height > 0:
            bottom_strip_for_extension = img.crop((0, target_content_height - strip_from_original_height, target_content_width, target_content_height))
            stretched_strip = bottom_strip_for_extension.resize((target_content_width, extended_area_height), Image.Resampling.BICUBIC)
            new_combined_img.paste(stretched_strip, (0, target_content_height))

        gradient_overlay_image = Image.new('RGBA', new_combined_img.size, (0, 0, 0, 0))
        draw_gradient = ImageDraw.Draw(gradient_overlay_image)

        gradient_top_y_on_canvas = target_content_height
        for y_relative_to_extended_area in range(extended_area_height):
            alpha = int(255 * (y_relative_to_extended_area / extended_area_height) * 0.95) 
            absolute_y_on_canvas = gradient_top_y_on_canvas + y_relative_to_extended_area
            draw_gradient.line([(0, absolute_y_on_canvas), (final_canvas_width, absolute_y_on_canvas)], fill=(0, 0, 0, alpha))
        
        img = Image.alpha_composite(new_combined_img, gradient_overlay_image)
        draw = ImageDraw.Draw(img)

        if BRANDING_LOGO_PATH and os.path.exists(BRANDING_LOGO_PATH):
            try:
                logo = Image.open(BRANDING_LOGO_PATH).convert("RGBA")
                logo_height = int(target_content_height * 0.08)
                logo_width = int(logo.width * (logo_height / logo.height))
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

                padding = int(target_content_width * 0.02)
                logo_x = target_content_width - logo_width - padding
                logo_y = padding 
                
                logo_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                logo_overlay.paste(logo, (logo_x, logo_y), logo)
                img = Image.alpha_composite(img, logo_overlay)
                logger.info("Branding logo applied successfully.")
            except Exception as e:
                logger.error(f"Error applying branding logo: {e}")

        selected_font = ImageFont.load_default()
        title_font_size = max(int(target_content_height * 0.035), 20)

        if DEFAULT_FONT_PATH:
            try:
                selected_font = ImageFont.truetype(DEFAULT_FONT_PATH, title_font_size)
            except (IOError, OSError) as e:
                logger.warning(f"Could not load specified font: {e}.")
        else:
            logger.warning("No default font path specified.")
        
        draw = ImageDraw.Draw(img) 

        max_text_width_for_title = int(target_content_width * 0.45)
        horizontal_padding_text = int(target_content_width * 0.02)

        def get_wrapped_text_lines(text, font, max_width):
            lines = []
            if not text: return lines
            words = text.split()
            if not words: return lines

            current_line = words[0]
            for word in words[1:]:
                test_line = f"{current_line} {word}".strip()
                try:
                    bbox = font.getbbox(test_line)
                    text_width = bbox[2] - bbox[0]
                except AttributeError:
                    text_width = font.getsize(test_line)[0]

                if text_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
            return lines

        wrapped_title_lines = get_wrapped_text_lines(title_text, selected_font, max_text_width_for_title)

        total_text_height_for_placement = 0
        line_height_list = []
        for line in wrapped_title_lines:
            try:
                bbox = selected_font.getbbox(line)
                line_height = bbox[3] - bbox[1]
                line_height_list.append(line_height + int(title_font_size * 0.2))
            except AttributeError:
                line_height_list.append(title_font_size + 3)
        
        total_text_height_for_placement = sum(line_height_list)

        bottom_align_y_coord = final_canvas_height - horizontal_padding_text 
        current_y_text_draw = bottom_align_y_coord - total_text_height_for_placement

        min_y_for_text = target_content_height + horizontal_padding_text 
        if current_y_text_draw < min_y_for_text:
            current_y_text_draw = min_y_for_text

        for i, line in enumerate(wrapped_title_lines):
            try:
                bbox = selected_font.getbbox(line)
                line_width = bbox[2] - bbox[0]
            except AttributeError:
                line_width = selected_font.getsize(line)[0]

            x_text_draw = target_content_width - horizontal_padding_text - line_width
            
            create_text_with_shadow(
                draw, (x_text_draw, current_y_text_draw), line, selected_font,
                (255, 255, 255, 255), (0, 0, 0, 180), (2, 2)
            current_y_text_draw += line_height_list[i]

        output_filename = f"{safe_filename}_{int(time.time())}.jpg"
        output_full_path = os.path.join(IMAGE_OUTPUT_FOLDER, output_category_folder, output_filename)

        final_img_for_save = img.convert('RGB')
        
        buffer = BytesIO()
        final_img_for_save.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        base64_data_uri = f"data:image/jpeg;base64,{base64_encoded_image}"

        with open(output_full_path, 'wb') as f:
            f.write(image_bytes)

        logger.info(f"Transformed image saved to disk: {output_full_path}")
        return output_full_path, base64_data_uri

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}")
        return None, None
    except IOError as e:
        logger.error(f"Error processing image: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during image transformation: {e}", exc_info=True)
        return None, None

def _gemini_generate_content_with_retry(model, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_INITIAL_RETRY_DELAY_SECONDS):
    """
    Helper function to call Gemini's generate_content with retry logic for transient errors.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if not response.text or response.text.strip() == "":
                logger.warning(f"Attempt {attempt + 1}: Gemini returned empty response. Retrying...")
                raise ValueError("Empty response from Gemini model.")

            return response
        except (
            exceptions.InternalServerError,
            exceptions.ResourceExhausted,
            exceptions.DeadlineExceeded,
            requests.exceptions.RequestException,
            ValueError
        ) as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Gemini API call failed after {max_retries} attempts.")
                raise

def perform_research_agent(target_topic, competitors):
    """
    Acts as the 'Research Agent'. Uses Gemini to find SEO keywords and outline suggestions.
    Outputs a JSON string.
    """
    if not RESEARCH_MODEL:
        logger.error("Research model not initialized. Skipping research agent.")
        return None

    prompt = (
        f"You are an expert SEO Keyword Research Agent specializing in market analysis and content strategy. "
        f"Your task is to perform comprehensive SEO keyword research and outline generation for the topic: '{target_topic}'.\n\n"
        f"Analyze content from top competitors (e.g., {', '.join(competitors[:5])}) to identify relevant SEO keywords, content gaps, and structural insights.\n\n"
        f"**Crucial:** Based on the topic, original source information, and keyword research, generate a **unique, catchy, and SEO-optimized blog post title (H1)** that will attract readers and rank well. This title should be distinct from the original source titles and reflect a consolidated, in-depth perspective.\n\n"
        "## Process Flow:\n"
        "1.  **Initial Keyword Discovery:** Identify primary (high search volume, high relevance) and secondary (long-tail) keyword clusters related to the target topic.\n"
        "2.  **Competitive Analysis:** Provide 2-3 key insights into competitor strategies and content gaps in relation to the topic.\n"
        "3.  **Keyword Evaluation:** Assess search volume and competition levels for identified keywords. Prioritize high-value, relevant keywords for SEO optimization.\n"
        "4.  **Outline Creation:** Generate a detailed, hierarchical blog post outline (using markdown headings `##`, `###`) that strategically incorporates the high-value keywords. Ensure the outline flows logically and covers comprehensive aspects of the topic. Suggest potential sub-sections for FAQs, case studies, or data points where appropriate.\n\n"
        "## Output Specifications:\n"
        "Generate a JSON object (as a string) with the following structure. Ensure the `blog_outline` is a valid markdown string.\n"
        "```json\n"
        "{{\n"
        "  \"suggested_blog_title\": \"Your Unique and Catchy Blog Post Title Here\",\n"
        "  \"primary_keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n"
        "  \"secondary_keywords\": {{\"sub_topic1\": [\"keywordA\", \"keywordB\"], \"sub_topic2\": [\"keywordC\", \"keywordD\"]}},\n"
        "  \"competitor_insights\": \"Summary of competitor strategies and content gaps.\",\n"
        "  \"blog_outline\": \"## Introduction\\n\\n### Hook\\n\\n## Main Section 1: [Section Title]\\n\\n### Sub-section 1.1\\n\\n## Conclusion\\n\"\n"
        "}}\n"
        "```\n"
        "**Constraints:** Focus on commercially relevant terms. Exclude branded competitor terms. The entire output must be a single, valid JSON string. The `blog_outline` must contain at least 8 distinct markdown headings (H2 or H3). The `suggested_blog_title` should be concise, impactful, and ideally under 70 characters. Do NOT include any introductory or concluding remarks outside the JSON block."
    )
    try:
        logger.info(f"Generating research for: '{target_topic[:70]}...'")
        response = _gemini_generate_content_with_retry(RESEARCH_MODEL, prompt)
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            research_data = json.loads(json_str)
            logger.info("Research generation successful.")
            return research_data
        else:
            logger.warning(f"Could not find valid JSON in markdown block. Attempting to parse raw response.")
            try:
                research_data = json.loads(response.text.strip())
                logger.info("Research generation successful (parsed raw response).")
                return research_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse research response as JSON. Raw response:\n{response.text[:500]}...")
                return None
            
    except (json.JSONDecodeError, ValueError, requests.exceptions.RequestException, exceptions.InternalServerError, exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
        logger.error(f"Research Agent generation failed: {e}.")
        return None
    except Exception as e:
        logger.error(f"Research Agent generation failed: {e}", exc_info=True)
        return None

def generate_content_agent(consolidated_article_data, research_output, transformed_image_filepath):
    """
    Acts as the 'Content Generator Agent'. Uses Gemini to write the blog post
    based on aggregated source data and research output.
    """
    if not CONTENT_MODEL:
        logger.error("Content model not initialized. Skipping content generation.")
        return None

    image_path_for_prompt = transformed_image_filepath if transformed_image_filepath else "None"

    primary_keywords_str = ', '.join(research_output.get('primary_keywords', []))
    secondary_keywords_str = ', '.join([kw for sub_list in research_output.get('secondary_keywords', {}).values() for kw in sub_list])

    new_blog_title = research_output.get('suggested_blog_title', consolidated_article_data.get('consolidated_topic', 'Default Consolidated Blog Title'))

    combined_content_for_prompt = consolidated_article_data.get('combined_content', '')
    if len(combined_content_for_prompt) > 4000:
        combined_content_for_prompt = combined_content_for_prompt[:4000] + "\n\n[...Content truncated for prompt brevity...]"
        logger.info(f"Truncated combined_content for prompt: {len(combined_content_for_prompt)} characters.")

    consolidated_article_data_for_prompt = consolidated_article_data.copy()
    consolidated_article_data_for_prompt['combined_content'] = combined_content_for_prompt

    prompt = (
        f"You are a specialized Blog Writing Agent that transforms SEO research and aggregated article data "
        f"into comprehensive, publication-ready, SEO-optimized blog posts. You excel at creating in-depth, "
        f"authoritative content by synthesizing information from multiple sources, while maintaining reader engagement and SEO best practices.\n\n"
        f"## Input Requirements:\n"
        f"1.  `aggregated_source_data`: {json.dumps(consolidated_article_data_for_prompt, indent=2)}\n"
        f"2.  `research_output`: {json.dumps(research_output, indent=2)}\n"
        f"3.  `transformed_image_path_info`: '{image_path_for_prompt}' (This is the file path to the main featured image. Do NOT embed this image again within the content body. It will be handled separately in the HTML template.)\n\n"
        f"## Content Specifications:\n"
        f"-   **Word Count:** Aim for 2500-3000 words. Synthesize and expand thoughtfully on the `aggregated_source_data['combined_content']`, adding depth, specific (even if fabricated) details, and related information from your training data. Do NOT simply copy-paste content from the input. Rewrite and integrate.\n"
        f"-   **Heading Structure:** Use the provided outline (`research_output['blog_outline']`). Ensure a minimum of 25 headings (`##` and `###` only, except for the main H1 title).\n"
        f"-   **Paragraph Length:** Each paragraph should contain at least 5 sentences for comprehensive coverage, unless it's a short intro/outro or a bullet point explanation.\n"
        f"-   **Writing Style:** Professional yet conversational, engaging, and human-like. Avoid jargon where simpler terms suffice. Do NOT mention that you are an AI or generated the content.\n"
        f"-   **Target Audience:** Broad audience interested in the specified category.\n"
        f"-   **Keyword Integration:** Naturally weave `primary_keywords` ({primary_keywords_str}) and `secondary_keywords` ({secondary_keywords_str}) throughout the text without keyword stuffing. Integrate them into headings, subheadings, and body paragraphs.\n"
        f"-   **Content Expansion:** Elaborate significantly on the `aggregated_source_data['combined_content']` by adding specific details, explanations, and context, drawing from your extensive knowledge base. Emphasize synthesizing information from *all provided sources* to create a unique and comprehensive article.\n"
        f"-   **Data & Examples:** Incorporate relevant data, statistics, and real-world examples (even if not in original article, create plausible ones). Invent realistic numbers, dates, names, and organizations where appropriate to enhance credibility. Ensure these details *support* the main points derived from the aggregated content.\n"
        f"-   **Linking:** Generate relevant external links where appropriate (e.g., `[Source Name](https://www.reputable-source.com/article-about-topic)`). **Crucially, ensure these are actual, plausible URLs from reputable domains related to the topic (e.g., 'nytimes.com/tech-news', 'theverge.com/reviews', 'medicaljournals.org/research', 'forbes.com/business-insights', 'espn.com/sports-analysis'). Invent these URLs realistically and embed them naturally within the surrounding sentences. Do NOT use the `@` symbol or any other prefix before links or raw URLs. Do NOT include `example.com` or similar placeholder domains.**\n"
        f"-   **Image Inclusion:** Do NOT include any markdown `![alt text](image_path)` syntax for the featured image within the generated content body. The featured image is handled separately.\n"
        f"## Output Structure:\n"
        f"Generate the complete blog post in markdown format. It must start with a metadata block followed by the blog content.\n\n"
        f"**Metadata Block (exact key-value pairs, no --- delimiters, newline separated):**\n"
        f"title: {new_blog_title}\n"
        f"description: {consolidated_article_data.get('combined_description', 'A comprehensive and insightful look at the latest news and trends.').replace('"', '')[:155]}\n"
        f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"categories: [{consolidated_article_data.get('category', 'general')}, {', '.join(research_output.get('primary_keywords', [])[:2])}]\n"
        f"tags: [{', '.join(research_output.get('primary_keywords', []) + research_output.get('secondary_keywords', {}).get(list(research_output.get('secondary_keywords', {}).keys())[0], []) if research_output.get('secondary_keywords') else research_output.get('primary_keywords', []))}]\n"
        f"featuredImage: {transformed_image_filepath if transformed_image_filepath else 'None'}\n\n"
        f"**Blog Content (following the metadata block):**\n"
        f"1.  **Main Title (H1):** Start with an H1 heading based on the provided `suggested_blog_title`. Example: `# {new_blog_title}`.\n"
        f"2.  **Introduction (2-3 paragraphs):** Hook the reader. Clearly state the problem or topic and your blog's value proposition.\n"
        f"3.  **Main Sections:** Follow the `blog_outline` from `research_output`. Expand each section (`##`) and sub-section (`###`). Ensure each section provides substantial information.\n"
        f"4.  **FAQ Section:** Include 5-7 frequently asked questions with detailed, comprehensive answers, related to the topic and incorporating keywords.\n"
        f"5.  **Conclusion:** Summarize key takeaways, provide a forward-looking statement, and a clear call-to-action.\n"
        f"Do NOT include any introductory or concluding remarks outside the blog content itself (e.g., 'Here is your blog post'). **Do NOT include any bracketed instructions (like `[mention this]`), placeholders (like `example.com`), or any comments intended for me within the output markdown. The entire output must be polished, final content, ready for publication.**"
    )
    try:
        logger.info(f"Generating full blog content for: '{new_blog_title[:70]}...'")
        response = _gemini_generate_content_with_retry(CONTENT_MODEL, prompt)
        
        content = response.text.strip()
        content = clean_ai_artifacts(content)
        
        logger.info("Content generation successful.")
        return content
        
    except (ValueError, requests.exceptions.RequestException, exceptions.InternalServerError, exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
        logger.error(f"Content Agent generation failed: {e}.")
        return None
    except Exception as e:
        logger.error(f"Content Agent generation failed: {e}", exc_info=True)
        return None

def clean_ai_artifacts(content):
    """Enhanced cleaning of AI-generated artifacts and placeholders."""
    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'\s*@\S+', '', content)
    
    placeholder_domains = [
        'example.com', 'example.org', 'placeholder.com', 'yoursite.com',
        'website.com', 'domain.com', 'site.com', 'yourblogname.com', 'ai-generated.com'
    ]
    for domain in placeholder_domains:
        content = re.sub(rf'\[[^\]]*\]\(https?://(?:www\.)?{re.escape(domain)}[^\)]*\)', '', content, flags=re.IGNORECASE)
        content = re.sub(rf'https?://(?:www\.)?{re.escape(domain)}\S*', '', content, flags=re.IGNORECASE)
    
    ai_patterns = [
        r'(?i)note:.*?(?=\n|$)',
        r'(?i)important:.*?(?=\n|$)',
        r'(?i)remember to.*?(?=\n|$)',
        r'(?i)please.*?(?=\n|$)',
        r'(?i)you should.*?(?=\n|$)',
        r'<!--.*?-->',
        r'/\*.*?\*/',
    ]
    for pattern in ai_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = '\n'.join([line.strip() for line in content.split('\n')])
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    return content.strip()

def parse_markdown_metadata(markdown_content):
    """
    Parses metadata from the top of a markdown string.
    Returns a dictionary of metadata and the remaining blog content.
    """
    metadata = {}
    lines = markdown_content.split('\n')
    content_start_index = 0

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            content_start_index = i + 1
            break
        if ':' in stripped_line:
            key, value = stripped_line.split(':', 1)
            metadata[key.strip()] = value.strip()
        else:
            content_start_index = i
            break
    else:
        content_start_index = len(lines)

    blog_content_only = '\n'.join(lines[content_start_index:]).strip()

    if blog_content_only.startswith('# '):
        h1_line_end = blog_content_only.find('\n')
        if h1_line_end != -1:
            h1_title = blog_content_only[2:h1_line_end].strip()
            if 'title' not in metadata:
                metadata['title'] = h1_title
            blog_content_only = blog_content_only[h1_line_end:].strip()
        else:
            h1_title = blog_content_only[2:].strip()
            if 'title' not in metadata:
                metadata['title'] = h1_title
            blog_content_only = ""

    return metadata, blog_content_only

def markdown_to_html(markdown_text, main_featured_image_filepath=None, main_featured_image_b64_data_uri=None):
    """
    Converts a subset of Markdown to HTML.
    """
    html_text = markdown_text
    html_text = clean_ai_artifacts(html_text)

    html_text = re.sub(r'###\s*(.*)', r'<h3>\1</h3>', html_text)
    html_text = re.sub(r'##\s*(.*)', r'<h2>\1</h2>', html_text)
    html_text = re.sub(r'#\s*(.*)', r'<h1>\1</h1>', html_text)

    html_text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', html_text)
    html_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'_(.*?)_', r'<em>\1</em>', html_text)
    html_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_text)

    html_text = re.sub(r'^\s*([-*]|\d+\.)\s+(.*)$', r'<li>\2</li>', html_text, flags=re.MULTILINE)
    
    def wrap_lists(match):
        list_items_html = match.group(0)
        if re.search(r'<li>\s*\d+\.', list_items_html):
            return f'<ol>{list_items_html}</ol>'
        else:
            return f'<ul>{list_items_html}</ul>'
            
    html_text = re.sub(r'(<li>.*?</li>\s*)+', wrap_lists, html_text, flags=re.DOTALL)

    def image_replacer(match):
        alt_text = match.group(1)
        src_url = match.group(2)
        
        if main_featured_image_filepath and os.path.basename(src_url) == os.path.basename(main_featured_image_filepath):
            logger.info(f"Replacing markdown image link with Base64 data URI.")
            return f'<img src="{main_featured_image_b64_data_uri}" alt="{alt_text}" class="in-content-image">'
        else:
            escaped_alt_text = alt_text.replace('"', '"')
            return f'<img src="{src_url}" alt="{escaped_alt_text}" class="in-content-image">'

    html_text = re.sub(r'!\[(.*?)\]\((.*?)\)', image_replacer, html_text)

    html_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', html_text)

    lines = html_text.split('\n')
    parsed_lines = []
    current_paragraph_lines = []

    block_tags_re = re.compile(r'^\s*<(h\d|ul|ol|li|img|a|div|p|blockquote|pre|table|script|style|br)', re.IGNORECASE)

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if current_paragraph_lines:
                para_content = ' '.join(current_paragraph_lines).strip()
                if para_content:
                    parsed_lines.append(f"<p>{para_content}</p>")
                current_paragraph_lines = []
            parsed_lines.append('')
        elif block_tags_re.match(stripped_line):
            if current_paragraph_lines:
                para_content = ' '.join(current_paragraph_lines).strip()
                if para_content:
                    parsed_lines.append(f"<p>{para_content}</p>")
                current_paragraph_lines = []
            parsed_lines.append(line)
        else:
            current_paragraph_lines.append(line)

    if current_paragraph_lines:
        para_content = ' '.join(current_paragraph_lines).strip()
        if para_content:
            parsed_lines.append(f"<p>{para_content}</p>")

    final_html_content = '\n'.join(parsed_lines)

    final_html_content = re.sub(r'<p>\s*</p>', '', final_html_content)
    final_html_content = re.sub(r'<p><br\s*/?></p>', '', final_html_content)
    final_html_content = re.sub(r'<h1>(.*?)</h1>', r'<h2>\1</h2>', final_html_content)

    return final_html_content

def generate_enhanced_html_template(title, description, keywords, image_url_for_seo, 
                                  image_src_for_html_body, html_blog_content, 
                                  category, article_url_for_disclaimer, published_date):
    """Generate enhanced HTML template with better styling and comprehensive SEO elements."""
    # Escape special characters for HTML attributes
    escaped_title = title.replace('"', '&quot;')
    escaped_description = description.replace('"', '&quot;')
    
    # Escape for JSON-LD separately
    json_ld_headline = title.replace('"', '\\"')
    json_ld_description = description.replace('"', '\\"')
    
    structured_data = f"""
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "NewsArticle",
      "headline": "{json_ld_headline}",
      "image": ["{image_url_for_seo}"],
      "datePublished": "{published_date}T00:00:00Z",
      "dateModified": "{published_date}T00:00:00Z",
      "articleSection": "{category.capitalize()}",
      "author": {{
        "@type": "Organization",
        "name": "AI Content Creator"
      }},
      "publisher": {{
        "@type": "Organization",
        "name": "Your Publication Name",
        "logo": {{
          "@type": "ImageObject",
          "url": "{image_url_for_seo}"
        }}
      }},
      "mainEntityOfPage": {{
        "@type": "WebPage",
        "@id": "{article_url_for_disclaimer}"
      }},
      "description": "{json_ld_description}"
    }}
    </script>
    """
    
    html_styles = """
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --text-color: #333;
            --light-bg: #f5f7fa;
            --card-bg: #ffffff;
            --border-color: #e0e0e0;
            --shadow-light: 0 4px 15px rgba(0,0,0,0.08);
            --shadow-hover: 0 6px 20px rgba(0,0,0,0.12);
        }

        body {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: var(--text-color);
            background: var(--light-bg);
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 850px;
            margin: 30px auto;
            padding: 25px;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: var(--shadow-light);
            transition: all 0.3s ease-in-out;
        }
        .container:hover {
            box-shadow: var(--shadow-hover);
        }

        .article-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .category-tag {
            display: inline-block;
            background: var(--primary-color);
            color: white;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.8px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 2.2em;
            color: var(--secondary-color);
            margin-bottom: 15px;
            line-height: 1.3;
        }
        h2 {
            font-size: 1.7em;
            color: var(--secondary-color);
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px dashed var(--border-color);
        }
        h3 {
            font-size: 1.3em;
            color: var(--secondary-color);
            margin-top: 25px;
            margin-bottom: 10px;
        }

        p {
            margin-bottom: 1.2em;
        }

        .featured-image {
            width: 100%;
            height: auto;
            max-height: 843.75px;
            object-fit: cover;
            border-radius: 8px;
            margin-top: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .in-content-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2em auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        a {
            color: var(--primary-color);
            text-decoration: none;
            transition: color 0.2s ease-in-out;
        }
        a:hover {
            color: #1a5e8c;
            text-decoration: underline;
        }

        ul, ol {
            margin-left: 25px;
            margin-bottom: 1.5em;
        }
        li {
            margin-bottom: 0.6em;
        }

        .source-link {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.95em;
            text-align: center;
            color: #666;
        }

        @media (max-width: 768px) {
            .container {
                margin: 15px;
                padding: 15px;
            }
            h1 { font-size: 1.8em; }
            h2 { font-size: 1.5em; }
            h3 { font-size: 1.2em; }
            .category-tag { font-size: 0.8em; padding: 6px 14px; }
        }
    </style>
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escaped_title}</title>
    <meta name="description" content="{escaped_description}">
    <meta name="keywords" content="{keywords}">
    <meta name="robots" content="index, follow">
    <meta name="author" content="AI Content Creator">
    
    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="article">
    <meta property="og:url" content="{article_url_for_disclaimer}">
    <meta property="og:title" content="{escaped_title}">
    <meta property="og:description" content="{escaped_description}">
    <meta property="og:image" content="{image_url_for_seo}">
    
    <!-- Twitter -->
    <meta property="twitter:card" content="summary_large_image">
    <meta property="twitter:url" content="{article_url_for_disclaimer}">
    <meta property="twitter:title" content="{escaped_title}">
    <meta property="twitter:description" content="{escaped_description}">
    <meta property="twitter:image" content="{image_url_for_seo}">
    
    {structured_data}
    {html_styles}
</head>
<body>
    <div class="container">
        <div class="article-header">
            <span class="category-tag">{category.upper()}</span>
            <h1>{title}</h1>
            {f'<img src="{image_src_for_html_body}" alt="{escaped_title}" class="featured-image">' if image_src_for_html_body else ''}
        </div>
        <div class="article-content">
            {html_blog_content}
        </div>
        <div class="source-link">
            <p><strong>Disclaimer:</strong> This article was generated by an AI content creation system, synthesizing information from multiple sources. It may contain fictional details and external links for illustrative purposes.</p>
            <p>A primary source contributing to this content can be found here: <a href="{article_url_for_disclaimer}" target="_blank" rel="noopener noreferrer">{article_url_for_disclaimer}</a></p>
        </div>
    </div>
</body>
</html>"""

def save_blog_post(consolidated_topic_for_fallback, generated_markdown_content, category, transformed_image_filepath, transformed_image_b64, primary_source_url):
    """
    Saves the generated blog post in an HTML file with SEO elements.
    """
    metadata, blog_content_only_markdown = parse_markdown_metadata(generated_markdown_content)

    title = metadata.get('title', consolidated_topic_for_fallback)
    description_fallback = f"A comprehensive look at the latest news in {category} related to '{title}'."
    description = metadata.get('description', description_fallback).replace('"', '')[:155]

    keywords_from_meta = metadata.get('tags', '').replace(', ', ',').replace(' ', '_')
    if not keywords_from_meta:
        keywords = ','.join([category, 'news', 'latest', sanitize_filename(title)[:30]])
    else:
        keywords = keywords_from_meta.lower()

    image_src_for_html_body = transformed_image_b64 if transformed_image_b64 else ''
    image_url_for_seo = transformed_image_filepath if transformed_image_filepath and transformed_image_filepath != 'None' else ''
    published_date = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))

    html_blog_content = markdown_to_html(
        blog_content_only_markdown,
        main_featured_image_filepath=transformed_image_filepath,
        main_featured_image_b64_data_uri=transformed_image_b64
    )

    safe_title_for_file = sanitize_filename(title)
    folder = os.path.join(BLOG_OUTPUT_FOLDER, category)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{safe_title_for_file}.html")

    final_html_output = generate_enhanced_html_template(
        title, description, keywords, image_url_for_seo, 
        image_src_for_html_body, html_blog_content, 
        category, primary_source_url, published_date
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_html_output)
    logger.info(f"✅ Saved blog post: {file_path}")

def main():
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return

    global_competitors = [
        "forbes.com", "reuters.com", "bloomberg.com", "theverge.com",
        "techcrunch.com", "healthline.com", "webmd.com", "espn.com",
        "investopedia.com", "zdnet.com", "cnet.com", "medicalnewstoday.com",
        "bbc.com/news", "cnn.com", "nytimes.com"
    ]

    for category in CATEGORIES:
        logger.info(f"\n--- Starting processing for category: [{category.upper()}] ---")
        raw_articles = fetch_gnews_articles(category, max_articles_to_fetch=NUM_SOURCE_ARTICLES_TO_AGGREGATE)

        if not raw_articles:
            logger.info(f"No raw articles fetched for {category}. Skipping category.")
            continue

        consolidated_data = aggregate_articles(raw_articles, category)
        if not consolidated_data:
            logger.error(f"Failed to aggregate articles for {category}. Skipping blog generation.")
            continue

        consolidated_topic = consolidated_data['consolidated_topic']
        consolidated_image_url = consolidated_data['image_url']
        consolidated_description = consolidated_data['combined_description']
        consolidated_content_for_ai = consolidated_data['combined_content']
        primary_source_url_for_disclaimer = consolidated_data['primary_source_url']
        effective_competitors = list(set(global_competitors + consolidated_data['competitors']))

        logger.info(f"\n  Starting workflow for consolidated topic: '{consolidated_topic[:70]}...'")

        transformed_image_filepath = None
        transformed_image_b64 = None
        if consolidated_image_url:
            safe_image_filename = sanitize_filename(consolidated_topic)
            transformed_image_filepath, transformed_image_b64 = transform_image(
                consolidated_image_url,
                consolidated_topic,
                category,
                category,
                safe_image_filename
            )

        try:
            consolidated_article_data_for_ai = {
                "consolidated_topic": consolidated_topic,
                "combined_description": consolidated_description,
                "combined_content": consolidated_content_for_ai,
                "category": category,
                "original_image_url_selected": consolidated_image_url
            }

            if GEMINI_API_KEY:
                research_output = perform_research_agent(consolidated_topic, effective_competitors)
                if not research_output:
                    logger.error(f"Failed to get research output. Skipping content generation.")
                    continue
                logger.info(f"  Research successful. Suggested Title: '{research_output.get('suggested_blog_title', 'N/A')}'")
                generated_blog_markdown = generate_content_agent(
                    consolidated_article_data_for_ai,
                    research_output,
                    transformed_image_filepath
                )
                if not generated_blog_markdown:
                    logger.error(f"Failed to generate blog content. Skipping save.")
                    continue
            else:
                logger.warning("GEMINI_API_KEY not set. Skipping AI content generation.")
                generated_blog_markdown = (
                    f"title: {consolidated_topic}\n"
                    f"description: {consolidated_description}\n"
                    f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
                    f"categories: [{category}]\n"
                    f"tags: [{category}, news]\n"
                    f"featuredImage: {transformed_image_filepath or 'None'}\n\n"
                    f"# {consolidated_topic}\n\n"
                    f"<p>This is a placeholder blog post because AI generation was skipped due to missing API key.</p>\n"
                    f"<p>Original aggregated content details (first 500 chars): {consolidated_content_for_ai[:500]}...</p>"
                )
                research_output = {"primary_keywords": [], "secondary_keywords": {}, "competitor_insights": "", "blog_outline": "", "suggested_blog_title": consolidated_topic}

            save_blog_post(
                consolidated_topic,
                generated_blog_markdown,
                category,
                transformed_image_filepath,
                transformed_image_b64,
                primary_source_url_for_disclaimer
            )

        except Exception as e:
            logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            time.sleep(30) 

if __name__ == '__main__':
    main()
