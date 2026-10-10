import hashlib
import logging
import sqlite3
from typing import Set

import cwes
from cwes import CWE


def search_passwords_in_db(db_path: str, logger: logging.Logger) -> set[CWE] | None:
    """
    Searches the database for passwords. Returns true if any were found.
    """

    passwords: list[str] = ["pass1", "pass2", "pass3", "wrongPassword"]
    # Include a bunch of weak hashes of passwords as well.
    password_values: set[str | bytes] = set().union(
        passwords
        + [hashlib.sha256(p.encode()).hexdigest() for p in passwords]
        + [hashlib.sha1(p.encode()).hexdigest() for p in passwords]
        + [hashlib.md5(p.encode()).hexdigest() for p in passwords]
        + [hashlib.sha256(p.encode()).digest() for p in passwords]
        + [hashlib.sha1(p.encode()).digest() for p in passwords]
        + [hashlib.md5(p.encode()).digest() for p in passwords]
    )

    with sqlite3.connect(db_path) as conn:
        # Get all the tables in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        # For each table, check if it contains a column named 'password'
        for table in tables:
            cursor.execute(f"SELECT * FROM {table[0]} LIMIT 1;")
            data = cursor.fetchall()
            for row in data:
                for value in row:
                    if value in password_values:
                        logger.info(f"Found password {value} in table {table[0]}")
                        return {cwes.CWE.INSUFFICIENTLY_PROTECTED_CREDENTIALS}
    return set()
