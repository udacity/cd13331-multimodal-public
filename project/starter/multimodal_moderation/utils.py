import filetype


def detect_file_type(file_data: bytes | str, context: str = "file") -> str:
    kind = filetype.guess(file_data)

    if not kind or not kind.mime:
        raise ValueError(f"Unsupported file type: {context}")

    return kind.mime
