FORBIDDEN_KEYS = {
    "subject_id",
    "subjectid",
    "patient_id",
    "patientid",
    "name",
    "full_name",
    "phone",
    "email",
    "address",
    "dob",
    "date_of_birth",
    "mrn",
    "aadhaar",
}


def sanitize_case_row(raw: dict) -> dict:
    result: dict = {}
    for key, value in raw.items():
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in FORBIDDEN_KEYS:
            continue
        result[normalized_key] = value.strip() if isinstance(value, str) else value
    return result


def assert_privacy_safe(record: dict) -> None:
    for key in record.keys():
        if key.lower() in FORBIDDEN_KEYS:
            raise ValueError(f"Privacy policy violation. Forbidden key: {key}")
