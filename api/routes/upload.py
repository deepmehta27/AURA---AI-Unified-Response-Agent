from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from api.schemas import AgentResponse
from api.dependencies import get_image_agent, get_audio_agent, validate_file_size
import shutil
import os
import tempfile

router = APIRouter()

@router.post("/analyze", response_model=AgentResponse)
async def analyze_file(
    file: UploadFile = File(...),
    query: str = Form(None),
    task_type: str = Form("auto"),
    image_agent = Depends(get_image_agent),
    audio_agent = Depends(get_audio_agent)
):
    tmp_path = None
    try:
        # Validate file size (50MB max)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        validate_file_size(file_size)
        
        # Reset file pointer after reading
        await file.seek(0)
        
        # 1. Identify file type
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        
        is_image = ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        is_audio = ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']

        if not (is_image or is_audio):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        # 2. Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 3. Route to appropriate agent
        result = {}
        
        if is_image:
            analysis_mode = "describe" if task_type == "auto" else task_type
            result = image_agent.process({
                "image_path": tmp_path,
                "query": query,
                "analysis_type": analysis_mode
            })
            
        elif is_audio:
            analysis_mode = "transcribe" if task_type == "auto" else task_type
            result = audio_agent.process({
                "audio_path": tmp_path,
                "query": query,
                "analysis_type": analysis_mode
            })

        # 4. Return response
        return AgentResponse(
            success=result["success"],
            response=result.get("response", ""),
            error=result.get("error"),
            metadata=result.get("metadata")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 5. Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
