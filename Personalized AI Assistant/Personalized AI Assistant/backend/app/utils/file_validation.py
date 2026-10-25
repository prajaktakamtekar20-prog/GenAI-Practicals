import os
from pathlib import Path

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
MAX_FILE_SIZE_MB = 10


def validate_upload_file(file) -> None:
    if file is None:
        raise ValueError('No file provided.')

    filename = file.filename or ''
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError('Unsupported file type. Please upload PDF, DOCX, or TXT files only.')

    file_size = 0
    try:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
    except Exception:
        file_size = 0

    if file_size == 0:
        raise ValueError('Uploaded file is empty.')

    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f'File exceeds the {MAX_FILE_SIZE_MB} MB limit.')
