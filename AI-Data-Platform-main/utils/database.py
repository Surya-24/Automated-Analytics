import psycopg2
from typing import Tuple, Optional, List, Dict, Any


class DatabaseUtil:

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = None

    def _get_connection(self):
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def schema_details(self, schema_name: str = "public") -> str:
        schema_info_context = f"Database Schema: {schema_name}\n"
        conn = self._get_connection()
        if not conn:
            return f"Error: Unable to connect to database using provided credentials."

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s;", (schema_name,))
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context += f"\nTable: {table_name}\n"

                # Adding Columns & Data Types
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;",
                    (schema_name, table_name)
                )
                columns_list = cursor.fetchall()
                for column in columns_list:
                    schema_info_context += f"  Column: {column[0]}, Data Type: {column[1]}\n"

                # Adding Sample Data (safe read-only limit)
                try:
                    cursor.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT 5;')
                    sample_data = cursor.fetchall()
                    schema_info_context += "  Sample Data:\n"
                    for row in sample_data:
                        schema_info_context += f"    {row}\n"
                except Exception as sample_err:
                    schema_info_context += f"  Sample Data: (unavailable: {sample_err})\n"

        except Exception as e:
            schema_info_context += f"Error fetching schema details: {e}"
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        return schema_info_context

    def execute_sql(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Executes a SELECT query safely, returning:
        (rows_as_dict_list, error_message)
        """
        conn = self._get_connection()
        if not conn:
            return None, "Database connection could not be established."

        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    row_dict = {}
                    for col_name, val in zip(columns, row):
                        # Convert decimals/datetimes to serializable types if needed
                        row_dict[col_name] = str(val) if not isinstance(val, (int, float, bool, type(None))) else val
                    results.append(row_dict)
                return results, None
            else:
                return [], None
        except Exception as e:
            conn.rollback()
            return None, str(e)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()