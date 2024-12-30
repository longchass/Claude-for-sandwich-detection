import os
import json
import pandas as pd
from pathlib import Path

class KnowledgeBase:
    def __init__(self):
        self.entries = []
    
    def add_entry(self, title, content, category=None, tags=None):
        """Add a new entry to the knowledge base"""
        entry = {
            'title': title,
            'content': content,
            'category': category,
            'tags': tags or []
        }
        self.entries.append(entry)
    
    def to_claude_format(self):
        """Format the knowledge base in a way that's optimal for Claude"""
        formatted_text = "<knowledge_base>\n"
        
        for entry in self.entries:
            formatted_text += f"<entry>\n"
            formatted_text += f"<title>{entry['title']}</title>\n"
            if entry['category']:
                formatted_text += f"<category>{entry['category']}</category>\n"
            if entry['tags']:
                formatted_text += f"<tags>{', '.join(entry['tags'])}</tags>\n"
            formatted_text += f"<content>\n{entry['content']}\n</content>\n"
            formatted_text += f"</entry>\n"
        
        formatted_text += "</knowledge_base>"
        return formatted_text
    
    def save_to_file(self, filename):
        """Save the formatted knowledge base to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.to_claude_format())

def load_from_text_file(kb, file_path, category=None):
    """Load content from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        title = Path(file_path).stem
        kb.add_entry(title=title, content=content, category=category)

def load_from_csv(kb, csv_file):
    """Load entries from a CSV file with blockchain transaction data"""
    # Load data and detect sandwich attacks
    df = pd.read_csv(csv_file)
    
    # Add sandwich attack detection
    df['is_sandwich'] = 0
    df['is_victim_trade'] = 0
    df['transaction_label'] = ''
    
    # Group by block number to find potential sandwich patterns
    for block_num, block_txs in df.groupby('block_number'):
        if len(block_txs) >= 3:  # Need at least 3 transactions for a sandwich
            # Sort by event index within block
            block_txs_sorted = block_txs.sort_values('evt_index')
            
            # Look for patterns where same address appears as front-runner and back-runner
            addresses = block_txs_sorted['tx_from'].values
            for i in range(len(addresses)-2):
                if addresses[i] == addresses[i+2]:  # Same address in position i and i+2
                    # Get the three consecutive transactions
                    three_txs = block_txs_sorted.iloc[i:i+3]
                    
                    # Check if all three transactions involve USDT-WBNB pair
                    if three_txs['token_pair'].str.contains('USDT-WBNB', case=False).all():
                        # Mark transactions
                        idx1 = three_txs.iloc[1].name  # Get the index of the middle transaction
                        idx0 = three_txs.iloc[0].name  # Get the index of the front-run
                        idx2 = three_txs.iloc[2].name  # Get the index of the back-run
                        
                        df.at[idx1, 'is_sandwich'] = 1
                        df.at[idx1, 'is_victim_trade'] = 1
                        df.at[idx0, 'transaction_label'] = 'front-run'
                        df.at[idx2, 'transaction_label'] = 'back-run'
    
    sandwich_count = len(df[df['is_sandwich'] == 1])
    print(f"Found {sandwich_count} sandwich attacks")
    print(f"Processing {len(df)} transactions...")
    
    for idx, row in df.iterrows():
        # Create a descriptive title
        title = f"Transaction: {row['token_pair']} on {row['project']}"
        
        # Create detailed content from the transaction data
        content = f"""
Blockchain: {row['blockchain']}
Project: {row['project']}
Block Number: {row['block_number']}
Block Time: {row['block_time']}
Token Pair: {row['token_pair']}
Tokens Exchanged: {row['token_bought_amount']} {row['token_bought_symbol']} for {row['token_sold_amount']} {row['token_sold_symbol']}
Transaction Hash: {row['tx_hash']}
From Address: {row['tx_from']}
To Address: {row['tx_to']}
Event Index: {row['evt_index']}
Is Sandwich: {'Yes' if row['is_sandwich'] == 1 else 'No'}
Transaction Role: {row['transaction_label'] if row['transaction_label'] else 'Normal Trade'}
        """.strip()
        
        # Add the entry with appropriate category and tags
        tags = [
            str(row['blockchain']),
            str(row['project']),
            str(row['token_pair']),
            'transaction',
            'sandwich_attack' if row['is_sandwich'] == 1 or row['transaction_label'] in ['front-run', 'back-run'] else 'normal_trade'
        ]
        
        kb.add_entry(
            title=title,
            content=content,
            category=f"{row['blockchain']}_{row['project']}",
            tags=tags
        )

def load_from_directory(kb, directory_path, category=None):
    """Load all text files from a directory"""
    directory = Path(directory_path)
    for file_path in directory.glob('*.txt'):
        load_from_text_file(kb, file_path, category)

def main():
    # Create a new knowledge base
    kb = KnowledgeBase()

    # Load blockchain transaction data
    print("Loading transaction data from CSV...")
    load_from_csv(kb, 'USDT-WBNB only.csv')

    # Save the knowledge base
    kb.save_to_file('claude_knowledge_base.txt')
    print("Knowledge base has been saved to 'claude_knowledge_base.txt'")

    # Print confirmation of number of entries
    print(f"\nProcessed {len(kb.entries)} transactions")
    
    # Print sandwich attack statistics
    sandwich_entries = [e for e in kb.entries if 'sandwich_attack' in e['tags']]
    print(f"Total sandwich attacks: {len(sandwich_entries)}")
    
    # Print unique projects involved in sandwich attacks
    projects = set(e['category'].split('_')[1] for e in sandwich_entries)
    print("\nProjects involved in sandwich attacks:")
    for project in sorted(projects):
        project_attacks = len([e for e in sandwich_entries if e['category'].split('_')[1] == project])
        print(f"- {project}: {project_attacks} attacks")

if __name__ == "__main__":
    main() 