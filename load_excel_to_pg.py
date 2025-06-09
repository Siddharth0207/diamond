# load_excel_to_pg.py
import pandas as pd
from sqlalchemy import create_engine

def load_excel_to_postgres():
    """
    Loads data from an Excel file (Diamond_Records_1000.xlsx) into a PostgreSQL database table named 'diamonds'.

    - Reads the Excel file using pandas.
    - Connects to a PostgreSQL database using SQLAlchemy.
    - Writes the DataFrame to the 'diamonds' table, replacing it if it already exists.
    - Prints a success message upon completion.
    """

    # Load Excel file
    df = pd.read_excel("Diamond_Records_1000.xlsx")  # Ensure diamonds.xlsx is in the same folder or provide full path

    # Create DB engine connection
    engine = create_engine(
        "postgresql+psycopg2://diamond_user:strongpassword@localhost:5432/diamond_db"
    )

    # Write dataframe to PostgreSQL table named 'diamonds'
    df.to_sql("diamonds", engine, if_exists="replace", index=False)
    print("Data loaded successfully!")

if __name__ == "__main__":
    load_excel_to_postgres()
