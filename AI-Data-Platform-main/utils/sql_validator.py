import sqlglot
from sqlglot import exp
from typing import Tuple, List

# Forbidden SQL Expression Types that modify database state or structure
FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Grant,
    exp.Command,
    exp.Set,
)

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "vacuum", "reindex", "execute",
    "pg_sleep", "pg_read_file", "pg_write_file", "copy"
}

def validate_sql_safety(query: str) -> Tuple[bool, str]:
    """
    Deterministically validates that a SQL query is strictly read-only using AST parsing (sqlglot).
    
    Returns:
        (is_safe: bool, reason: str)
    """
    if not query or not query.strip():
        return False, "SQL query is empty."
    
    cleaned_query = query.strip().rstrip(";")
    
    # Check for multiple chained statements
    statements = [s for s in cleaned_query.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple SQL statements (chained queries) are prohibited for safety."
        
    try:
        parsed_expressions = sqlglot.parse(cleaned_query, read="postgres")
    except Exception as e:
        return False, f"SQL syntax parsing failed: {str(e)}"
    
    if not parsed_expressions or len(parsed_expressions) == 0:
        return False, "Unable to parse SQL query structure."
    
    for parsed in parsed_expressions:
        if parsed is None:
            continue
            
        # Verify root query expression is Select or Union
        if not isinstance(parsed, (exp.Select, exp.Union)):
            return False, f"Prohibited SQL statement type: {parsed.__class__.__name__}. Only read-only SELECT queries are allowed."
            
        # Recursively search AST for forbidden sub-expressions (e.g. DML/DDL inside CTEs or subqueries)
        for forbidden in FORBIDDEN_EXPRESSIONS:
            if parsed.find(forbidden):
                return False, f"Prohibited operation detected in AST: {forbidden.__name__}. Only read-only data querying is permitted."
                
        # Check all table/identifier/function names against dangerous keyword list
        for func in parsed.find_all(exp.Anonymous):
            if func.name.lower() in FORBIDDEN_KEYWORDS:
                return False, f"Potentially hazardous function execution detected: '{func.name}'"
                
    return True, "Query passed deterministic AST validation."
