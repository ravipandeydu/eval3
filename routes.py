import os
from fastapi import APIRouter, File, UploadFile

from schemas import QuestionRequest
from rag import get_rag_answer, reload_rag_system

router = APIRouter()


# List processed documents
@router.get("/documents")
def list_documents():
    try:
        # List all files in the docs folder
        documents = os.listdir("docs")
        return {"documents": documents}
    except Exception as e:
        return {"error": f"Failed to list documents: {str(e)}"}


# Upload and process documents
@router.post("/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create docs directory if it doesn't exist
        os.makedirs("docs", exist_ok=True)

        # Save the file to the docs folder
        with open(f"docs/{file.filename}", "wb") as f:
            content = await file.read()
            f.write(content)

        # Reload the RAG system to include the new document
        reload_success = reload_rag_system()

        if reload_success:
            return {
                "filename": file.filename,
                "message": "File uploaded and RAG system updated successfully",
            }
        else:
            return {
                "filename": file.filename,
                "message": "File uploaded but failed to update RAG system",
                "warning": "The new document may not be available for queries yet",
            }
    except Exception as e:
        return {"error": f"Failed to upload file: {str(e)}"}


# RAG queries
@router.post("/query")
async def get_answer_endpoint(request: QuestionRequest):
    # Process the question using RAG
    answer = get_rag_answer(request.question)
    return {"question": request.question, "answer": answer}
