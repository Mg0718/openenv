"""
Task data generator for the Data Clean Environment.

Generates dirty datasets and their golden (clean) references for 3 tasks:
- easy: fix_missing_values (20 rows, 5 columns)
- medium: dedup_and_normalize (50 rows, 8 columns)
- hard: full_pipeline (100 rows, 12 columns)
"""

import json
import copy
import random
from typing import Any, Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────
# DatasetCorruptor (procedural errors)
# ─────────────────────────────────────────────────────────────────────
class DatasetCorruptor:
    """Procedurally inject errors into a clean dataset."""
    ERROR_TYPES = ["missing", "duplicate", "wrong_date_fmt", "wrong_casing", "invalid_numeric"]
    
    def corrupt(self, golden: List[Dict[str, Any]], seed: int, error_rate: float = 0.3) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        dirty = []
        for row in golden:
            dirty.append(copy.deepcopy(row))
            # Duplication error chance
            if rng.random() < error_rate * 0.1:
                dirty.append(copy.deepcopy(row))
        
        for idx, row in enumerate(dirty):
            for col, val in row.items():
                if col == "id" or val is None:
                    continue
                if rng.random() < error_rate:
                    err = rng.choice(self.ERROR_TYPES)
                    self._apply_error(row, col, err, rng)
        return dirty
        
    def _apply_error(self, row: Dict[str, Any], col: str, err: str, rng: random.Random):
        val = row[col]
        if err == "missing":
            row[col] = None
        elif err == "wrong_date_fmt" and isinstance(val, str) and "-" in val:
            # simple swap for date format (assume ISO YYYY-MM-DD -> DD/MM/YYYY)
            parts = str(val).split("-")
            if len(parts) == 3:
                row[col] = f"{parts[2]}/{parts[1]}/{parts[0]}"
        elif err == "wrong_casing" and isinstance(val, str):
            row[col] = val.lower() if val.istitle() else val.upper()
        elif err == "invalid_numeric" and isinstance(val, (int, float)):
            row[col] = -abs(val) if val > 0 else 0


# ─────────────────────────────────────────────────────────────────────
# TASK 1: EASY — Fix Missing Values
# Employee records with ~30% missing values
# ─────────────────────────────────────────────────────────────────────

EASY_GOLDEN = [
    {"id": 1, "name": "Alice Johnson", "age": 29, "department": "Engineering", "salary": 75000},
    {"id": 2, "name": "Bob Smith", "age": 34, "department": "Marketing", "salary": 62000},
    {"id": 3, "name": "Carol Davis", "age": 27, "department": "Engineering", "salary": 80000},
    {"id": 4, "name": "David Lee", "age": 41, "department": "Sales", "salary": 55000},
    {"id": 5, "name": "Eva Martinez", "age": 36, "department": "HR", "salary": 58000},
    {"id": 6, "name": "Frank Wilson", "age": 31, "department": "Engineering", "salary": 78000},
    {"id": 7, "name": "Grace Kim", "age": 28, "department": "Marketing", "salary": 61000},
    {"id": 8, "name": "Henry Brown", "age": 45, "department": "Sales", "salary": 52000},
    {"id": 9, "name": "Iris Chen", "age": 33, "department": "Engineering", "salary": 82000},
    {"id": 10, "name": "Jack Taylor", "age": 38, "department": "HR", "salary": 60000},
    {"id": 11, "name": "Kate Adams", "age": 26, "department": "Marketing", "salary": 59000},
    {"id": 12, "name": "Leo Garcia", "age": 42, "department": "Sales", "salary": 57000},
    {"id": 13, "name": "Mia Robinson", "age": 30, "department": "Engineering", "salary": 76000},
    {"id": 14, "name": "Noah Clark", "age": 35, "department": "HR", "salary": 63000},
    {"id": 15, "name": "Olivia White", "age": 29, "department": "Marketing", "salary": 64000},
    {"id": 16, "name": "Paul Harris", "age": 37, "department": "Sales", "salary": 56000},
    {"id": 17, "name": "Quinn Lewis", "age": 32, "department": "Engineering", "salary": 79000},
    {"id": 18, "name": "Ruby Walker", "age": 40, "department": "HR", "salary": 61000},
    {"id": 19, "name": "Sam Hall", "age": 28, "department": "Marketing", "salary": 60000},
    {"id": 20, "name": "Tina Young", "age": 43, "department": "Sales", "salary": 54000},
]


def _make_easy_dirty() -> List[Dict[str, Any]]:
    """Create dirty version of easy dataset with ~30% missing values."""
    dirty = copy.deepcopy(EASY_GOLDEN)
    # Introduce missing values (None)
    missing_cells = [
        (1, "age"), (2, "salary"), (4, "department"), (5, "name"),
        (7, "age"), (8, "salary"), (9, "department"), (11, "age"),
        (12, "name"), (13, "salary"), (15, "department"), (16, "age"),
        (17, "salary"), (19, "department"),
    ]
    for row_idx, col in missing_cells:
        dirty[row_idx][col] = None
    return dirty


# ─────────────────────────────────────────────────────────────────────
# TASK 2: MEDIUM — Dedup & Normalize
# Customer records with duplicates, inconsistent dates, phone formats
# ─────────────────────────────────────────────────────────────────────

MEDIUM_GOLDEN = [
    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "phone": "+1-555-0101", "signup_date": "2024-01-15", "city": "New York", "plan": "premium", "active": True},
    {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "phone": "+1-555-0102", "signup_date": "2024-02-20", "city": "Los Angeles", "plan": "basic", "active": True},
    {"id": 3, "name": "Carol Davis", "email": "carol@example.com", "phone": "+1-555-0103", "signup_date": "2024-03-10", "city": "Chicago", "plan": "premium", "active": False},
    {"id": 4, "name": "David Lee", "email": "david@example.com", "phone": "+1-555-0104", "signup_date": "2024-01-25", "city": "Houston", "plan": "enterprise", "active": True},
    {"id": 5, "name": "Eva Martinez", "email": "eva@example.com", "phone": "+1-555-0105", "signup_date": "2024-04-01", "city": "Phoenix", "plan": "basic", "active": True},
    {"id": 6, "name": "Frank Wilson", "email": "frank@example.com", "phone": "+1-555-0106", "signup_date": "2024-02-14", "city": "Philadelphia", "plan": "premium", "active": False},
    {"id": 7, "name": "Grace Kim", "email": "grace@example.com", "phone": "+1-555-0107", "signup_date": "2024-05-05", "city": "San Antonio", "plan": "basic", "active": True},
    {"id": 8, "name": "Henry Brown", "email": "henry@example.com", "phone": "+1-555-0108", "signup_date": "2024-03-18", "city": "San Diego", "plan": "enterprise", "active": True},
    {"id": 9, "name": "Iris Chen", "email": "iris@example.com", "phone": "+1-555-0109", "signup_date": "2024-06-22", "city": "Dallas", "plan": "premium", "active": True},
    {"id": 10, "name": "Jack Taylor", "email": "jack@example.com", "phone": "+1-555-0110", "signup_date": "2024-01-30", "city": "San Jose", "plan": "basic", "active": False},
    {"id": 11, "name": "Kate Adams", "email": "kate@example.com", "phone": "+1-555-0111", "signup_date": "2024-07-12", "city": "Austin", "plan": "premium", "active": True},
    {"id": 12, "name": "Leo Garcia", "email": "leo@example.com", "phone": "+1-555-0112", "signup_date": "2024-04-08", "city": "Jacksonville", "plan": "basic", "active": True},
    {"id": 13, "name": "Mia Robinson", "email": "mia@example.com", "phone": "+1-555-0113", "signup_date": "2024-08-19", "city": "Fort Worth", "plan": "enterprise", "active": False},
    {"id": 14, "name": "Noah Clark", "email": "noah@example.com", "phone": "+1-555-0114", "signup_date": "2024-02-28", "city": "Columbus", "plan": "premium", "active": True},
    {"id": 15, "name": "Olivia White", "email": "olivia@example.com", "phone": "+1-555-0115", "signup_date": "2024-09-03", "city": "Charlotte", "plan": "basic", "active": True},
    {"id": 16, "name": "Paul Harris", "email": "paul@example.com", "phone": "+1-555-0116", "signup_date": "2024-05-17", "city": "Indianapolis", "plan": "premium", "active": True},
    {"id": 17, "name": "Quinn Lewis", "email": "quinn@example.com", "phone": "+1-555-0117", "signup_date": "2024-03-22", "city": "San Francisco", "plan": "enterprise", "active": False},
    {"id": 18, "name": "Ruby Walker", "email": "ruby@example.com", "phone": "+1-555-0118", "signup_date": "2024-10-11", "city": "Seattle", "plan": "basic", "active": True},
    {"id": 19, "name": "Sam Hall", "email": "sam@example.com", "phone": "+1-555-0119", "signup_date": "2024-06-30", "city": "Denver", "plan": "premium", "active": True},
    {"id": 20, "name": "Tina Young", "email": "tina@example.com", "phone": "+1-555-0120", "signup_date": "2024-04-25", "city": "Washington", "plan": "basic", "active": False},
    {"id": 21, "name": "Uma Patel", "email": "uma@example.com", "phone": "+1-555-0121", "signup_date": "2024-11-05", "city": "Nashville", "plan": "enterprise", "active": True},
    {"id": 22, "name": "Victor Nguyen", "email": "victor@example.com", "phone": "+1-555-0122", "signup_date": "2024-07-19", "city": "Oklahoma City", "plan": "premium", "active": True},
    {"id": 23, "name": "Wendy Scott", "email": "wendy@example.com", "phone": "+1-555-0123", "signup_date": "2024-05-08", "city": "Las Vegas", "plan": "basic", "active": True},
    {"id": 24, "name": "Xavier Brooks", "email": "xavier@example.com", "phone": "+1-555-0124", "signup_date": "2024-12-01", "city": "Portland", "plan": "premium", "active": False},
    {"id": 25, "name": "Yuki Tanaka", "email": "yuki@example.com", "phone": "+1-555-0125", "signup_date": "2024-08-14", "city": "Memphis", "plan": "enterprise", "active": True},
    {"id": 26, "name": "Zara Ahmed", "email": "zara@example.com", "phone": "+1-555-0126", "signup_date": "2024-06-03", "city": "Louisville", "plan": "basic", "active": True},
    {"id": 27, "name": "Aaron Price", "email": "aaron@example.com", "phone": "+1-555-0127", "signup_date": "2024-09-27", "city": "Baltimore", "plan": "premium", "active": True},
    {"id": 28, "name": "Bella Foster", "email": "bella@example.com", "phone": "+1-555-0128", "signup_date": "2024-03-05", "city": "Milwaukee", "plan": "basic", "active": False},
    {"id": 29, "name": "Carlos Rivera", "email": "carlos@example.com", "phone": "+1-555-0129", "signup_date": "2024-10-20", "city": "Albuquerque", "plan": "enterprise", "active": True},
    {"id": 30, "name": "Diana Moore", "email": "diana@example.com", "phone": "+1-555-0130", "signup_date": "2024-07-01", "city": "Tucson", "plan": "premium", "active": True},
]


def _make_medium_dirty() -> List[Dict[str, Any]]:
    """Create dirty version with duplicates, inconsistent dates & phones."""
    dirty = copy.deepcopy(MEDIUM_GOLDEN)

    # Add duplicate rows (with slight variations to make it realistic)
    duplicates = [
        {"id": 31, "name": "alice johnson", "email": "alice@example.com", "phone": "5550101", "signup_date": "01/15/2024", "city": "new york", "plan": "Premium", "active": True},
        {"id": 32, "name": "BOB SMITH", "email": "bob@example.com", "phone": "(555) 010-2", "signup_date": "20/02/2024", "city": "los angeles", "plan": "Basic", "active": True},
        {"id": 33, "name": "carol davis", "email": "carol@example.com", "phone": "555.0103", "signup_date": "Mar 10, 2024", "city": "CHICAGO", "plan": "PREMIUM", "active": False},
        {"id": 34, "name": "David Lee", "email": "david@example.com", "phone": "1-555-0104", "signup_date": "2024/01/25", "city": "houston", "plan": "Enterprise", "active": True},
        {"id": 35, "name": "Eva Martinez", "email": "eva@example.com", "phone": "+15550105", "signup_date": "April 1, 2024", "city": "PHOENIX", "plan": "BASIC", "active": True},
    ]
    dirty.extend(duplicates)

    # Inconsistent date formats on existing rows
    date_messes = {
        2: "02/20/2024",    # MM/DD/YYYY
        5: "04-01-2024",    # MM-DD-YYYY
        8: "Jun 22, 2024",  # Mon DD, YYYY
        11: "2024/04/08",   # YYYY/MM/DD
        14: "Sept 3, 2024", # Mon D, YYYY
        17: "10-11-2024",   # MM-DD-YYYY
        20: "2024/11/05",   # YYYY/MM/DD
        23: "12/01/2024",   # MM/DD/YYYY
    }
    for idx, bad_date in date_messes.items():
        dirty[idx]["signup_date"] = bad_date

    # Inconsistent phone formats
    phone_messes = {
        1: "5550101",       # no formatting
        4: "(555) 010-5",   # weird format
        7: "555.0108",      # dots
        10: "1-555-0111",   # no +
        13: "+15550114",    # no dashes
        16: "555 0117",     # spaces  
        19: "(555)0120",    # partial
        22: "555-0123",     # partial
    }
    for idx, bad_phone in phone_messes.items():
        dirty[idx]["phone"] = bad_phone

    # Inconsistent city capitalization
    city_messes = {0: "new york", 3: "HOUSTON", 6: "san antonio", 9: "SAN JOSE", 12: "fort worth", 15: "INDIANAPOLIS"}
    for idx, bad_city in city_messes.items():
        dirty[idx]["city"] = bad_city

    # Inconsistent plan casing
    plan_messes = {0: "PREMIUM", 3: "Enterprise", 6: "BASIC", 9: "Basic", 12: "enterprise"}
    for idx, bad_plan in plan_messes.items():
        dirty[idx]["plan"] = bad_plan

    return dirty


# ─────────────────────────────────────────────────────────────────────
# TASK 3: HARD — Full Pipeline
# Product inventory with ALL issue types combined
# ─────────────────────────────────────────────────────────────────────

HARD_GOLDEN = [
    {"id": 1, "product_name": "Wireless Mouse", "category": "electronics", "price": 29.99, "stock": 150, "supplier": "TechCorp", "sku": "WM-001", "weight_kg": 0.12, "rating": 4.5, "review_count": 230, "launch_date": "2023-06-15", "status": "active"},
    {"id": 2, "product_name": "USB-C Cable", "category": "electronics", "price": 12.99, "stock": 500, "supplier": "CableCo", "sku": "UC-002", "weight_kg": 0.05, "rating": 4.2, "review_count": 890, "launch_date": "2023-03-10", "status": "active"},
    {"id": 3, "product_name": "Office Chair", "category": "furniture", "price": 249.99, "stock": 45, "supplier": "ComfortSeats", "sku": "OC-003", "weight_kg": 15.0, "rating": 4.7, "review_count": 156, "launch_date": "2023-01-20", "status": "active"},
    {"id": 4, "product_name": "Desk Lamp", "category": "furniture", "price": 34.99, "stock": 200, "supplier": "BrightLight", "sku": "DL-004", "weight_kg": 1.2, "rating": 4.0, "review_count": 78, "launch_date": "2023-08-05", "status": "active"},
    {"id": 5, "product_name": "Mechanical Keyboard", "category": "electronics", "price": 89.99, "stock": 120, "supplier": "KeyMaster", "sku": "MK-005", "weight_kg": 0.95, "rating": 4.8, "review_count": 445, "launch_date": "2023-04-12", "status": "active"},
    {"id": 6, "product_name": "Monitor Stand", "category": "furniture", "price": 45.99, "stock": 80, "supplier": "DeskPro", "sku": "MS-006", "weight_kg": 3.5, "rating": 4.3, "review_count": 112, "launch_date": "2023-07-22", "status": "active"},
    {"id": 7, "product_name": "Webcam HD", "category": "electronics", "price": 59.99, "stock": 95, "supplier": "VisionTech", "sku": "WH-007", "weight_kg": 0.18, "rating": 4.1, "review_count": 234, "launch_date": "2023-02-28", "status": "active"},
    {"id": 8, "product_name": "Notebook A5", "category": "stationery", "price": 8.99, "stock": 1000, "supplier": "PaperWorks", "sku": "NA-008", "weight_kg": 0.25, "rating": 4.6, "review_count": 567, "launch_date": "2023-05-18", "status": "active"},
    {"id": 9, "product_name": "Pen Set", "category": "stationery", "price": 15.99, "stock": 300, "supplier": "WriteRight", "sku": "PS-009", "weight_kg": 0.15, "rating": 4.4, "review_count": 321, "launch_date": "2023-09-01", "status": "active"},
    {"id": 10, "product_name": "Headphones", "category": "electronics", "price": 149.99, "stock": 75, "supplier": "AudioMax", "sku": "HP-010", "weight_kg": 0.35, "rating": 4.9, "review_count": 678, "launch_date": "2023-11-10", "status": "active"},
    {"id": 11, "product_name": "Ergonomic Mouse Pad", "category": "furniture", "price": 19.99, "stock": 250, "supplier": "ComfortSeats", "sku": "EP-011", "weight_kg": 0.3, "rating": 4.1, "review_count": 89, "launch_date": "2023-06-30", "status": "active"},
    {"id": 12, "product_name": "USB Hub", "category": "electronics", "price": 24.99, "stock": 180, "supplier": "TechCorp", "sku": "UH-012", "weight_kg": 0.08, "rating": 4.3, "review_count": 145, "launch_date": "2023-10-05", "status": "active"},
    {"id": 13, "product_name": "Sticky Notes", "category": "stationery", "price": 5.99, "stock": 800, "supplier": "PaperWorks", "sku": "SN-013", "weight_kg": 0.1, "rating": 4.5, "review_count": 432, "launch_date": "2023-04-20", "status": "active"},
    {"id": 14, "product_name": "Standing Desk", "category": "furniture", "price": 399.99, "stock": 30, "supplier": "DeskPro", "sku": "SD-014", "weight_kg": 25.0, "rating": 4.8, "review_count": 201, "launch_date": "2023-08-15", "status": "active"},
    {"id": 15, "product_name": "Power Strip", "category": "electronics", "price": 18.99, "stock": 350, "supplier": "PowerPlus", "sku": "PS-015", "weight_kg": 0.45, "rating": 4.2, "review_count": 167, "launch_date": "2023-12-01", "status": "active"},
    {"id": 16, "product_name": "Whiteboard", "category": "stationery", "price": 39.99, "stock": 60, "supplier": "WriteRight", "sku": "WB-016", "weight_kg": 2.0, "rating": 4.0, "review_count": 78, "launch_date": "2023-07-10", "status": "active"},
    {"id": 17, "product_name": "Cable Organizer", "category": "electronics", "price": 11.99, "stock": 400, "supplier": "CableCo", "sku": "CO-017", "weight_kg": 0.2, "rating": 4.4, "review_count": 256, "launch_date": "2023-03-25", "status": "active"},
    {"id": 18, "product_name": "Desk Organizer", "category": "furniture", "price": 27.99, "stock": 150, "supplier": "DeskPro", "sku": "DO-018", "weight_kg": 1.5, "rating": 4.6, "review_count": 134, "launch_date": "2023-09-18", "status": "active"},
    {"id": 19, "product_name": "Laptop Stand", "category": "electronics", "price": 49.99, "stock": 110, "supplier": "TechCorp", "sku": "LS-019", "weight_kg": 1.8, "rating": 4.7, "review_count": 289, "launch_date": "2023-05-30", "status": "active"},
    {"id": 20, "product_name": "Marker Set", "category": "stationery", "price": 12.99, "stock": 220, "supplier": "WriteRight", "sku": "MS-020", "weight_kg": 0.3, "rating": 4.3, "review_count": 187, "launch_date": "2023-11-25", "status": "active"},
    {"id": 21, "product_name": "Wireless Charger", "category": "electronics", "price": 35.99, "stock": 160, "supplier": "PowerPlus", "sku": "WC-021", "weight_kg": 0.22, "rating": 4.5, "review_count": 345, "launch_date": "2023-02-14", "status": "active"},
    {"id": 22, "product_name": "Filing Cabinet", "category": "furniture", "price": 129.99, "stock": 25, "supplier": "ComfortSeats", "sku": "FC-022", "weight_kg": 18.0, "rating": 4.0, "review_count": 56, "launch_date": "2023-06-08", "status": "active"},
    {"id": 23, "product_name": "Bluetooth Speaker", "category": "electronics", "price": 39.99, "stock": 200, "supplier": "AudioMax", "sku": "BS-023", "weight_kg": 0.5, "rating": 4.6, "review_count": 423, "launch_date": "2023-10-30", "status": "active"},
    {"id": 24, "product_name": "Paper Clips", "category": "stationery", "price": 3.99, "stock": 1500, "supplier": "PaperWorks", "sku": "PC-024", "weight_kg": 0.05, "rating": 4.1, "review_count": 678, "launch_date": "2023-01-05", "status": "active"},
    {"id": 25, "product_name": "Surge Protector", "category": "electronics", "price": 29.99, "stock": 140, "supplier": "PowerPlus", "sku": "SP-025", "weight_kg": 0.6, "rating": 4.4, "review_count": 198, "launch_date": "2023-08-28", "status": "active"},
    {"id": 26, "product_name": "Bookshelf", "category": "furniture", "price": 89.99, "stock": 35, "supplier": "DeskPro", "sku": "BK-026", "weight_kg": 12.0, "rating": 4.2, "review_count": 87, "launch_date": "2023-04-15", "status": "active"},
    {"id": 27, "product_name": "Stapler", "category": "stationery", "price": 9.99, "stock": 450, "supplier": "WriteRight", "sku": "ST-027", "weight_kg": 0.35, "rating": 4.0, "review_count": 234, "launch_date": "2023-12-20", "status": "active"},
    {"id": 28, "product_name": "Docking Station", "category": "electronics", "price": 79.99, "stock": 65, "supplier": "TechCorp", "sku": "DS-028", "weight_kg": 0.4, "rating": 4.7, "review_count": 312, "launch_date": "2023-07-05", "status": "active"},
    {"id": 29, "product_name": "Desk Mat", "category": "furniture", "price": 22.99, "stock": 180, "supplier": "ComfortSeats", "sku": "DM-029", "weight_kg": 0.8, "rating": 4.5, "review_count": 156, "launch_date": "2023-09-12", "status": "active"},
    {"id": 30, "product_name": "Planner", "category": "stationery", "price": 14.99, "stock": 300, "supplier": "PaperWorks", "sku": "PL-030", "weight_kg": 0.4, "rating": 4.3, "review_count": 289, "launch_date": "2023-11-01", "status": "active"},
]


def _make_hard_dirty() -> List[Dict[str, Any]]:
    """Create dirty version with ALL issue types combined."""
    dirty = copy.deepcopy(HARD_GOLDEN)

    # 1. Missing values
    missing_cells = [
        (1, "supplier"), (3, "weight_kg"), (5, "rating"),
        (7, "category"), (9, "stock"), (11, "supplier"),
        (13, "weight_kg"), (15, "rating"), (17, "category"),
        (19, "stock"), (21, "supplier"), (23, "weight_kg"),
        (25, "rating"), (27, "category"), (29, "stock"),
    ]
    for row_idx, col in missing_cells:
        dirty[row_idx][col] = None

    # 2. Duplicate rows (slightly varied)
    duplicates = [
        {"id": 31, "product_name": "wireless mouse", "category": "Electronics", "price": 29.99, "stock": 150, "supplier": "techcorp", "sku": "WM-001", "weight_kg": 0.12, "rating": 4.5, "review_count": 230, "launch_date": "06/15/2023", "status": "Active"},
        {"id": 32, "product_name": "USB-C CABLE", "category": "ELECTRONICS", "price": 12.99, "stock": 500, "supplier": "CABLECO", "sku": "UC-002", "weight_kg": 0.05, "rating": 4.2, "review_count": 890, "launch_date": "03-10-2023", "status": "ACTIVE"},
        {"id": 33, "product_name": "Office chair", "category": "Furniture", "price": 249.99, "stock": 45, "supplier": "comfortseats", "sku": "OC-003", "weight_kg": 15.0, "rating": 4.7, "review_count": 156, "launch_date": "Jan 20, 2023", "status": "active"},
        {"id": 34, "product_name": "DESK LAMP", "category": "FURNITURE", "price": 34.99, "stock": 200, "supplier": "BrightLight", "sku": "DL-004", "weight_kg": 1.2, "rating": 4.0, "review_count": 78, "launch_date": "2023/08/05", "status": "Active"},
        {"id": 35, "product_name": "mechanical keyboard", "category": "electronics", "price": 89.99, "stock": 120, "supplier": "keymaster", "sku": "MK-005", "weight_kg": 0.95, "rating": 4.8, "review_count": 445, "launch_date": "April 12, 2023", "status": "ACTIVE"},
    ]
    dirty.extend(duplicates)

    # 3. Invalid values
    dirty[0]["price"] = -29.99     # negative price
    dirty[2]["stock"] = -10        # negative stock
    dirty[4]["rating"] = 5.5       # rating > 5
    dirty[6]["price"] = 0          # zero price
    dirty[8]["rating"] = -1.0      # negative rating
    dirty[10]["review_count"] = -5 # negative review

    # 4. Inconsistent date formats
    date_messes = {
        0: "06/15/2023", 2: "01-20-2023", 4: "Apr 12, 2023",
        6: "2023/02/28", 8: "05-18-2023", 10: "June 30, 2023",
        12: "04/20/2023", 14: "12-01-2023", 16: "03/25/2023",
        18: "May 30, 2023", 20: "02/14/2023", 22: "Oct 30, 2023",
        24: "08-28-2023", 26: "2023/12/20", 28: "Sept 12, 2023",
    }
    for idx, bad_date in date_messes.items():
        dirty[idx]["launch_date"] = bad_date

    # 5. Inconsistent category casing
    cat_messes = {0: "Electronics", 2: "FURNITURE", 4: "electronics",
                  6: "Electronics", 8: "Stationery", 10: "FURNITURE",
                  12: "Stationery", 14: "ELECTRONICS", 16: "electronics"}
    for idx, bad_cat in cat_messes.items():
        dirty[idx]["category"] = bad_cat

    # 6. Inconsistent status casing
    status_messes = {0: "Active", 2: "ACTIVE", 4: "Active",
                     6: "ACTIVE", 8: "Active", 10: "ACTIVE"}
    for idx, bad_status in status_messes.items():
        dirty[idx]["status"] = bad_status

    # 7. Semantic / Contextual errors
    dirty[1]["price"] = 1299.0          # Magnitude outlier (cents vs dollars)
    dirty[3]["weight_kg"] = 1200.0      # Unit inconsistency (grams instead of kg)
    dirty[5]["launch_date"] = "2050-01-01" # Future launch date
    dirty[7]["rating"] = 4.6
    dirty[7]["review_count"] = 0        # Contradiction: positive rating but 0 reviews

    return dirty


# ─────────────────────────────────────────────────────────────────────
# TASK 4: ML IMPACT — Downstream Reward
# ─────────────────────────────────────────────────────────────────────

def _generate_ml_data(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    data = []
    for i in range(1, n + 1):
        age = rng.uniform(20, 70)
        salary = rng.uniform(30000, 120000)
        balance = rng.uniform(0, 200000)
        credit_score = rng.uniform(300, 850)
        score = (age/70)*0.2 + (salary/120000)*0.3 + (credit_score/850)*0.2 + (balance/200000)*0.3
        target = 1 if score > 0.52 + rng.uniform(-0.1, 0.1) else 0
        data.append({
            "id": i,
            "age": int(age),
            "salary": int(salary),
            "credit_score": int(credit_score),
            "balance": round(balance, 2),
            "purchased": target
        })
    return data

ML_IMPACT_GOLDEN = _generate_ml_data(100, seed=123)
ML_IMPACT_TEST_DATA = _generate_ml_data(50, seed=456)

def _make_ml_impact_dirty(seed: int) -> List[Dict[str, Any]]:
    corruptor = DatasetCorruptor()
    return corruptor.corrupt(ML_IMPACT_GOLDEN, seed=seed, error_rate=0.2)

# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

TASKS = {
    "fix_missing_values": {
        "description": "Fix missing values in an employee dataset. ~30% of cells are missing. Fill them appropriately based on context.",
        "difficulty": "easy",
        "dirty_data": None,  # lazily generated
        "golden_data": None,
    },
    "dedup_and_normalize": {
        "description": "Clean a customer dataset with duplicate rows (varied casing/format), inconsistent date formats (MM/DD, DD/MM, etc.), inconsistent phone formats, and inconsistent text casing. Deduplicate and normalize all formats.",
        "difficulty": "medium",
        "dirty_data": None,
        "golden_data": None,
    },
    "full_pipeline": {
        "description": "Complete data cleaning pipeline on a product inventory. Issues include: missing values, duplicate rows, invalid values (negative prices, ratings > 5), inconsistent date formats, inconsistent category/status casing. Fix ALL issues to produce a clean dataset.",
        "difficulty": "hard",
        "dirty_data": None,
        "golden_data": None,
    },
    "ml_impact": {
        "description": "Clean customer financial dataset to improve downstream ML classification. Remove NaN, outliers, negative salaries.",
        "difficulty": "hard",
        "dirty_data": None,
        "golden_data": None,
    },
}


def get_task_data(task_name: str, seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Get dirty and golden data for a task.

    Returns:
        (dirty_data, golden_data) tuple
    """
    # Use procedural generation if seed is provided
    # Keep legacy hardcoded tasks for now but allow ML_impact with seed
    seed = seed or 42
    
    if task_name == "fix_missing_values":
        return _make_easy_dirty(), copy.deepcopy(EASY_GOLDEN)
    elif task_name == "dedup_and_normalize":
        return _make_medium_dirty(), copy.deepcopy(MEDIUM_GOLDEN)
    elif task_name == "full_pipeline":
        return _make_hard_dirty(), copy.deepcopy(HARD_GOLDEN)
    elif task_name == "ml_impact":
        return _make_ml_impact_dirty(seed=seed), copy.deepcopy(ML_IMPACT_GOLDEN)
    else:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")


def get_task_description(task_name: str) -> str:
    """Get the description for a task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    return TASKS[task_name]["description"]


def list_tasks() -> List[str]:
    """List all available task names."""
    return list(TASKS.keys())
