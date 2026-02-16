import asyncio
from fastapi import UploadFile
import aiofiles
from stt import transcribe
from io import BytesIO
import os  # Add this!

async def test_with_file(file_path):
    """Test with an existing audio file"""
    print(f"📁 Testing with file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return


    # Read file first
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()

    # THEN create UploadFile with ALL arguments
    from io import BytesIO
    upload_file = UploadFile(
        filename=os.path.basename(file_path),  # Argument 1 ✅
        file=BytesIO(content),  # Argument 2 ✅
        content_type="audio/mpeg"  # Argument 3 ✅
    )
    # Transcribe
    print("⏳ Transcribing...")
    result = await transcribe(upload_file)

    print(f"\n📝 Text: {result.get('text', '')}")
    print(f"🌐 Language: {result.get('language_name', 'Unknown')}")

    return result


if __name__ == "__main__":
    # Test with your MP3 file
    file_path = "Nazki.mp3"  # or Obama.mp3
    asyncio.run(test_with_file(file_path))