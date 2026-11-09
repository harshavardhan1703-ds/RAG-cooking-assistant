import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag_project_impl import VectorStore, EmbeddingManager, RAGRetriever, rag_simple

# Load environment variables
load_dotenv()

def run_evaluation():
    print("Initializing RAG system components for evaluation...")
    
    # 1. Initialize backend components (matching your UI setup)
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vectorstore = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
    retriever = RAGRetriever(vectorstore, embedding_manager)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in the .env file.")
        return
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=4096
    )

    # 2. Load the 25 evaluation test questions
    test_questions = [
        { "id": 1, "type": "direct", "question": "What is the exact baking temperature and preheating time specified for the classic chocolate chip cookies?" },
        { "id": 2, "type": "direct", "question": "How much butter and brown sugar are required for the pancake batter recipe?" },
        { "id": 3, "type": "direct", "question": "What is the primary thickening agent used in the creamy mushroom soup recipe?" },
        { "id": 4, "type": "direct", "question": "What specific herbs are listed in the marinade for the grilled chicken skewers?" },
        { "id": 5, "type": "direct", "question": "How long should the sourdough bread dough rest during its first bulk fermentation?" },
        { "id": 6, "type": "direct", "question": "What is the recommended internal cooking temperature for medium-rare beef steak according to the guide?" },
        { "id": 7, "type": "direct", "question": "What liquids are combined to make the base of the tomato basil pasta sauce?" },
        { "id": 8, "type": "direct", "question": "At what stage of cooking should garlic be added to avoid burning it in the stir-fry recipe?" },
        { "id": 9, "type": "direct", "question": "What equipment or pan size is recommended for baking the lasagna recipe?" },
        { "id": 10, "type": "direct", "question": "How many eggs are needed to make a standard batch of vanilla sponge cake?" },
        { "id": 11, "type": "multi_chunk", "question": "What are all the ingredients and corresponding steps required to make the homemade pizza dough from scratch?" },
        { "id": 12, "type": "multi_chunk", "question": "Can you summarize the complete preparation and cooking process for making chicken biryani using the retrieved recipe chunks?" },
        { "id": 13, "type": "multi_chunk", "question": "What ingredients go into the curry paste, and how do they combine with coconut milk in the Thai green curry recipe?" },
        { "id": 14, "type": "multi_chunk", "question": "What is the step-by-step procedure for preparing the pastry crust and assembling the apple pie filling?" },
        { "id": 15, "type": "multi_chunk", "question": "How do you prepare the vegetables and cook them alongside the protein in the beef stroganoff recipe?" },
        { "id": 16, "type": "multi_chunk", "question": "What are the storage guidelines and reheating instructions provided for the vegetable lasagna?" },
        { "id": 17, "type": "multi_chunk", "question": "How should the dry rubs be mixed and applied before smoking the pork ribs according to the documents?" },
        { "id": 18, "type": "multi_chunk", "question": "What are the precise steps for boiling pasta and finishing it in the carbonara sauce without scrambling the eggs?" },
        { "id": 19, "type": "negative", "question": "Can I substitute heavy cream with motor oil in the creamy mushroom soup recipe?" },
        { "id": 20, "type": "negative", "question": "How do I bake a gluten-free chocolate soufflé using ingredients that are not mentioned in the recipe database?" },
        { "id": 21, "type": "negative", "question": "What is the recipe for authentic Japanese ramen noodles?" },
        { "id": 22, "type": "negative", "question": "Can I cook chicken at room temperature indefinitely without refrigeration according to the cookbook guidelines?" },
        { "id": 23, "type": "negative", "question": "How many calories are in a slice of chocolate cake if the nutritional data is completely missing from the text?" },
        { "id": 24, "type": "negative", "question": "What is the procedure for canning homemade tomato sauce for 5-year shelf storage based on the documents?" },
        { "id": 25, "type": "negative", "question": "Can I use steel cookware in a microwave oven safely according to the recipe safety notes?" }
    ]

    results = []
    print(f"\nRunning evaluation across {len(test_questions)} test questions...\n" + "-"*50)

    # 3. Execute queries and capture answers
    for item in test_questions:
        q_id = item["id"]
        q_type = item["type"]
        question = item["question"]
        
        print(f"[{q_id}/25] [{q_type.upper()}] Evaluating: {question}")
        
        try:
            # Calls your exact rag_simple function from your backend module
            answer = rag_simple(question, retriever, llm, top_k=3)
            results.append({
                "id": q_id,
                "type": q_type,
                "question": question,
                "answer": answer,
                "grounded": None  # Fill in True/False after manual or automated check
            })
            print(f"-> Generated response length: {len(answer)} chars\n")
        except Exception as e:
            print(f"-> Error generating response: {e}\n")

    # 4. Save results to a JSON file for your records
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("-" * 50)
    print(f"Evaluation complete! Results saved to '{output_file}'.")

if __name__ == "__main__":
    run_evaluation()