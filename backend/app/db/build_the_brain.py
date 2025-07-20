
import os
import pymupdf  
from langchain.text_splitter import RecursiveCharacterTextSplitter # To smartly chunk text
import chromadb # Our vector database ("smart filing cabinet")
from dotenv import load_dotenv # To securely load our API key
import google.generativeai as genai # To use Google's Gemini models
from PIL import Image # To handle image files
import time # To handle potential API rate limits

# --- 2. CONFIGURATION: All our settings in one place ---

# -- File Paths --
PDF_FILE_PATH = "assets/oxford.pdf"  # Updated to correct path
IMAGE_OUTPUT_FOLDER = "images" # Folder to save extracted pictures
DB_PATH = "my_medical_db" # Folder where the final "brain" will be stored

# -- AI Model Names --
EMBEDDING_MODEL = "models/text-embedding-004" # For creating "meaning coordinates"
VISION_MODEL = "models/gemini-1.5-pro-latest" # For "seeing" and describing images

# -- Database Settings --
DB_COLLECTION_NAME = "oxford_multimodal" # The name of our table inside the database

# --- 3. THE BLUEPRINT: Our functions (The Robot's Jobs) ---

def initialize_database():
    """Sets up the ChromaDB client and creates our collection."""
    print("Setting up the database...")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create or get collection
    try:
        collection = client.create_collection(
            name=DB_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {DB_COLLECTION_NAME}")
    except Exception:
        collection = client.get_collection(name=DB_COLLECTION_NAME)
        print(f"Using existing collection: {DB_COLLECTION_NAME}")
    
    return collection

def extract_text_from_pdf(pdf_path):
    """
    Job 1: The Librarian.
    Reads the PDF and extracts all text, page by page.
    Returns a list of dictionaries, each with page text and metadata.
    """
    print('Starting to extract text from pdf...')
    all_text = []

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text() 
            if text: 
                all_text.append({
                    "page_content": text,
                    "metadata": {"source": "Oxford Handbook", "page": page.number + 1}
                })

    print(f'Finished extracting text from pdf, the total number of pages is {len(all_text)}')
    return all_text

def extract_images_from_pdf(pdf_path, output_folder):
    """
    Job 2: The Art Curator.
    Finds all images in the PDF and saves them to a folder.
    Returns a list of file paths to the saved images.
    """
    print(f"Extracting images from PDF to {output_folder}...")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = []
    
    with pymupdf.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get the XREF of the image
                xref = img[0]
                
                # Extract the image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Create filename
                image_filename = f"page_{page_num + 1}_img_{img_index}.png"
                image_path = os.path.join(output_folder, image_filename)
                
                # Save the image
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                image_paths.append(image_path)
    
    print(f"Extracted {len(image_paths)} images")
    return image_paths

def create_text_cards(text_data):
    """
    Job 3: The Text Indexer.
    Takes the raw text and splits it into small, meaningful chunks.
    Returns a list of these "text cards".
    """
    print("Creating text cards from extracted text...")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Combine all text data
    all_texts = []
    for item in text_data:
        all_texts.append({
            "content": item["page_content"],
            "metadata": item["metadata"]
        })
    
    # Split the texts
    text_cards = []
    for item in all_texts:
        # Split the content
        chunks = text_splitter.split_text(item["content"])
        
        # Create cards for each chunk
        for i, chunk in enumerate(chunks):
            text_cards.append({
                "content": chunk,
                "metadata": {
                    **item["metadata"],
                    "chunk_id": i,
                    "type": "text"
                }
            })
    
    print(f"Created {len(text_cards)} text cards")
    return text_cards

def create_image_cards(image_paths):
    """
    Job 4: The AI Art Critic.
    Uses Gemini Vision to create detailed text descriptions for each image.
    Returns a list of "image cards," each containing the AI's description and metadata.
    """
    print("Creating image cards with AI descriptions...")
    
    # Initialize the vision model
    vision_model = genai.GenerativeModel(VISION_MODEL)
    
    image_cards = []
    
    for image_path in image_paths:
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Create a medical-focused prompt
            prompt = """You are a medical expert analyzing a medical image or diagram. 
            Please provide a detailed description of what you see, including:
            - Any anatomical structures visible
            - Medical conditions or symptoms shown
            - Diagnostic information or measurements
            - Any text, labels, or captions in the image
            - The type of medical image (X-ray, diagram, chart, etc.)
            
            Be precise and use medical terminology where appropriate."""
            
            # Get AI description
            response = vision_model.generate_content([prompt, image])
            description = response.text
            
            # Create image card
            image_cards.append({
                "content": description,
                "metadata": {
                    "source": "Oxford Handbook",
                    "image_path": image_path,
                    "type": "image"
                }
            })
            
            print(f"Processed image: {image_path}")
            
            # Add delay to respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    print(f"Created {len(image_cards)} image cards")
    return image_cards

def store_cards_in_database(collection, cards, card_type):
    """
    Job 5: The Final Filer.
    Takes a list of cards (text or image), creates their "meaning coordinates" (embeddings),
    and stores them permanently in our database.
    """
    print(f"Storing {len(cards)} {card_type} cards in database...")
    
    # Process in batches to avoid API limits
    batch_size = 100
    
    for i in range(0, len(cards), batch_size):
        batch = cards[i:i + batch_size]
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for j, card in enumerate(batch):
            documents.append(card["content"])
            metadatas.append(card["metadata"])
            ids.append(f"{card_type}_{i + j}")
        
        # Add to collection (ChromaDB will handle embedding generation)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored batch {i//batch_size + 1} of {card_type} cards")
        
        # Add delay between batches
        time.sleep(0.5)
    
    print(f"Successfully stored all {card_type} cards")


# --- 4. THE MAIN EVENT: Running the entire offline plan ---

if __name__ == "__main__":
    print(">>> STARTING OFFLINE BUILD PROCESS: Creating the AI's 'Brain' <<<")

    # Load the secret API key from the .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("Please create a .env file with your Google API key.")
        exit(1)
    
    genai.configure(api_key=api_key)

    # Step 1: Set up the database
    db_collection = initialize_database()

    # --- Text Processing Pipeline ---
    print("\n--- Processing TEXT from the PDF ---")
    raw_text_data = extract_text_from_pdf(PDF_FILE_PATH)
    text_cards = create_text_cards(raw_text_data)
    store_cards_in_database(db_collection, text_cards, card_type="text")

    # --- Image Processing Pipeline ---
    print("\n--- Processing IMAGES from the PDF ---")
    image_file_paths = extract_images_from_pdf(PDF_FILE_PATH, IMAGE_OUTPUT_FOLDER)
    image_cards = create_image_cards(image_file_paths)
    store_cards_in_database(db_collection, image_cards, card_type="image")

    print("\n>>> OFFLINE BUILD PROCESS COMPLETE! <<<")
    print(f"The 'brain' is ready and stored in the '{DB_PATH}' folder.")
    print(f"Total items in the knowledge base: {db_collection.count()}") 