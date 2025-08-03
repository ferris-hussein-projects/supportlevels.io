
#!/usr/bin/env python3
"""
Database migration script to add missing columns to user_settings table
"""
import os
import psycopg2
from urllib.parse import urlparse

def migrate_database():
    """Add missing columns to user_settings table"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("DATABASE_URL environment variable not found")
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check if columns exist and add them if they don't
        migrations = [
            ("support_threshold", "ALTER TABLE user_settings ADD COLUMN support_threshold REAL DEFAULT 0.1"),
            ("resistance_threshold", "ALTER TABLE user_settings ADD COLUMN resistance_threshold REAL DEFAULT 0.1"),
            ("level_type", "ALTER TABLE user_settings ADD COLUMN level_type VARCHAR(20) DEFAULT 'support'")
        ]
        
        for column_name, alter_sql in migrations:
            # Check if column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='user_settings' AND column_name=%s
            """, (column_name,))
            
            if not cur.fetchone():
                print(f"Adding column {column_name}...")
                cur.execute(alter_sql)
                print(f"Successfully added column {column_name}")
            else:
                print(f"Column {column_name} already exists")
        
        # Commit changes
        conn.commit()
        print("Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_database()
