import sqlite3
import pandas as pd
import numpy as np

bd = r"C:\Bureau\Proyectos\Vaccines\nextflow\Docker\gnn_epitope\data\db_epitopes.db"
table = "epitopes"
list_columns = [
    "Results",
    "Method description",
    "Epitope Type",
    "Epitope Length",
    "Positive ratio",
    "Number of tests",
    "Quantitative Measure",
    "Units",
    "Success ratio",
    "Epitope Relation",
    "Immunoglobulin Domain",
    "Antibody Name"
]

id_col = "Epitope Sequence"
id_value = "A570"


def global_features(bd, table, list_columns, id_col, id_value):

    conn = sqlite3.connect(bd)
    query = f"""
                SELECT {', '.join(f'"{col}"' for col in list_columns)}
                FROM {table}
                WHERE "{id_col}" = ?
                LIMIT 1
            """

    df = pd.read_sql_query(query, conn, params=(id_value,))
    conn.close()

    if df.empty:
        print(f"No se encontraron datos para {id_value} en la tabla {table}.")
        return None

    row = df.iloc[0].to_dict()

    normalized = {}
    # Normalización personalizada
    for col in list_columns:
        if col.lower() == "results":
            normalized[col] = 2
            pass

        elif col.lower() == "method description":
            print(col.lower(), "->", "Method description")
            normalized[col] = 3
            pass

        elif col.lower() == "epitope type":
            normalized[col] = 1
            pass

        elif col.lower() == "epitope length":
            normalized[col] = 0
            pass

        elif col.lower() == "positive ratio":
            normalized[col] = 0
            pass

        elif col.lower() == "number of tests":
            normalized[col] = 1
            pass

        elif col.lower() == "quantitative measure":
            normalized[col] = 0
            pass

        elif col.lower() == "units":
            normalized[col] = 4
            pass

        elif col.lower() == "success ratio":
            normalized[col] = 0
            pass

        elif col.lower() == "epitope relation":
            normalized[col] = 2
            pass

        elif col.lower() == "immunoglobulin domain":
            normalized[col] = 3
            pass

        elif col.lower() == "antibody name":
            normalized[col] = 4
            pass

    print("Valores normalizados:")
    for key in row.keys():
        if key in normalized:
            print(f"  - {key}: {row[key]} -> {normalized[key]}")
            row[key] = normalized[key]
    print(f"Características globales para {id_value}: {row}")
    vector_values = [row[col] for col in list_columns]

    return np.array(vector_values)

if __name__ == "__main__":
    features = global_features(bd, table, list_columns, id_col, id_value)
    print("Vector de características globales:", features)