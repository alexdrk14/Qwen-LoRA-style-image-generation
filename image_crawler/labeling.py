import os 
import json
import torch
import code
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from pathlib import Path
import hashlib
from tqdm import tqdm

ROOT_DIR = os.path.dirname('/'.join(os.path.abspath(__file__).split('/')[:-1]))
import base64
import requests

def describe_image_with_ollama(prompt: str, image_path: str, model: str = "gemma3:4b") -> str:
    """

    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prepare request
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64]
    }

    # Send to local Ollama API
    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
    response.raise_for_status()

    # Collect streamed output
    output_text = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                # Extract text content (basic JSON parsing)
                part = data.split('"response":"')[1].split('"')[0]
                output_text += part

    return output_text.strip()

class DatasetPreparator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize captioning model
        print("Loading captioning model...")
        #self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        #self.model = InstructBlipForConditionalGeneration.from_pretrained(
        #    "Salesforce/instructblip-vicuna-7b",
        #    torch_dtype=torch.float16,
        #device_map="auto"
        #)
        self.prompt = """Analyze this image and create a concise training caption (50-75 words) for image generation fine-tuning.

REQUIRED FORMAT - Output ONLY the final caption, nothing else:

[ARTIST_STYLE], [subject and scene], [brushwork style], [color palette with specific colors], [line quality], [lighting description], [composition type], [texture/medium], [2-3 distinctive style characteristics], [mood]

RULES:
- Start with trigger word: [ARTIST_STYLE]In the provided example, use "SimonStalenhagStyle"
- Write as ONE continuous sentence with comma separations
- Be specific with colors (e.g., "muted olive greens and warm siennas" not just "green and brown")
- Focus on visual style elements, not interpretations
- Use concise descriptive phrases
- NO bullet points, NO explanations, NO structured sections
- Output ONLY the caption text

EXAMPLE OUTPUT:
"SimonStalenhagStyle, dense forest landscape with clearing and winding path, loose expressive brushwork with visible layered strokes, muted color palette of moss greens olive tones and deep browns, delicate flowing linework defining foliage details, diffuse overcast lighting with soft even shadows, elevated perspective with strong atmospheric depth, digital painting technique with realistic texture, hyper-detailed foliage rendering and melancholic tranquil atmosphere"

"""
        
        # Aspect ratio buckets for efficient training
        self.aspect_buckets = {
            "square": (1024, 1024),      # 1:1
            "portrait": (768, 1344),      # 4:7
            "landscape": (1344, 768),     # 7:4
            "wide": (1536, 640),          # 12:5
            "tall": (640, 1536),          # 5:12
        }
    
    def get_aspect_bucket(self, width, height):
        """Determine which aspect ratio bucket an image belongs to"""
        ratio = width / height
        return 'square'
        
        if 0.9 <= ratio <= 1.1:
            return "square"
        elif ratio < 0.7:
            return "tall"
        elif ratio < 0.9:
            return "portrait"
        elif ratio > 1.4:
            return "wide"
        else:
            return "landscape"

    def resize_image(self, image, target_size):
        """
        Resize and crop image to fill target size (maintains aspect ratio, no padding).
        """
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Calculate scale to cover the target area (may crop)
        scale = max(target_w / img_w, target_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        # Resize while maintaining aspect ratio
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Calculate crop box (center crop)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        image = image.crop((left, top, right, bottom))

        return image

    def generate_caption(self, image):
        """Generate caption using InstructBLIP"""
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=800,
                num_beams=3,
                temperature=0.7,
                do_sample=True
            )
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        code.interact(local=dict(**globals(), **locals()))
        # Remove the prompt from the beginning of the caption
        caption = caption.replace(self.prompt, "").strip()
        return caption
    
    def process_image(self, image_path, custom_caption=None):
        """Process a single image: resize, generate caption, save"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Determine aspect bucket
            bucket = self.get_aspect_bucket(*original_size)
            target_size = self.aspect_buckets[bucket]
            
            # Resize image
            processed_image = self.resize_image(image, target_size)

            # Create unique filename based on original path
            original_name = Path(image_path).stem
            file_hash = hashlib.md5(str(image_path).encode()).hexdigest()[:8]
            new_filename = f"{original_name}_{file_hash}"
            # Save processed image
            output_image_path = self.output_dir / f"{new_filename}.jpg"
            processed_image.save(output_image_path, "JPEG", quality=95)
            
            # Generate caption if not provided
            if custom_caption is None:
                caption = describe_image_with_ollama(prompt= self.prompt, 
                                                     image_path= output_image_path) #self.generate_caption(image)
            else:
                caption = custom_caption

            return {
                "filename": f"{new_filename}.jpg",
                "caption": caption
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_dataset(self, input_dir, custom_captions=None):
        """Process entire dataset"""
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f"**/*{ext}")))
            image_files.extend(list(input_path.glob(f"**/*{ext.upper()}")))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process images
        bucket_counts = {bucket: 0 for bucket in self.aspect_buckets.keys()}
        
        for image_path in tqdm(image_files, desc="Processing images"):
            custom_caption = None
            if custom_captions and str(image_path) in custom_captions:
                custom_caption = custom_captions[str(image_path)]
            
            result = self.process_image(image_path, custom_caption)
            if result:
                with open(self.output_dir / result['filename'].replace('.jpg', '.txt'), 'a') as f:
                    f.write(result['caption'])
        
        # Print statistics
        print(f"\nProcessing complete!")
        print(f"Output directory: {self.output_dir}")
        print("\nAspect ratio distribution:")
        for bucket, count in bucket_counts.items():
            print(f"  {bucket}: {count} images")

def make_caption_labels(output_dir=ROOT_DIR + "/processed_dataset", input_dir=ROOT_DIR + "/images"):
    # Process dataset
    preparator = DatasetPreparator(output_dir)
    preparator.process_dataset(input_dir)
