import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class SandwichPatternAnalyzer:
    def __init__(self):
        self.df = None
        self.sandwich_df = None
        self.normal_df = None
    
    def load_data(self, csv_file):
        """Load and preprocess the transaction data"""
        print("Loading transaction data...")
        self.df = pd.read_csv(csv_file)
        
        # Convert is_sandwich to int if it's not already
        if self.df['is_sandwich'].dtype != 'int64':
            self.df['is_sandwich'] = self.df['is_sandwich'].astype(int)
        
        # Split into sandwich and normal transactions
        self.sandwich_df = self.df[self.df['is_sandwich'] == 1]
        self.normal_df = self.df[self.df['is_sandwich'] == 0]
        
        print(f"Total transactions: {len(self.df)}")
        print(f"Sandwich attacks: {len(self.sandwich_df)}")
        print(f"Normal trades: {len(self.normal_df)}")
    
    def analyze_token_patterns(self):
        """Analyze patterns in token pairs and individual tokens"""
        print("\nToken Pair Analysis:")
        
        # Clean token pairs (remove NaN values)
        self.df['token_pair'] = self.df['token_pair'].fillna('UNKNOWN')
        self.sandwich_df['token_pair'] = self.sandwich_df['token_pair'].fillna('UNKNOWN')
        
        # Most attacked token pairs
        sandwich_pairs = self.sandwich_df['token_pair'].value_counts()
        print("\nMost Targeted Token Pairs:")
        print(sandwich_pairs.head(10))
        
        # Attack rate by token pair
        pair_stats = pd.DataFrame({
            'total_trades': self.df['token_pair'].value_counts(),
            'sandwich_attacks': self.sandwich_df['token_pair'].value_counts()
        }).fillna(0)
        
        pair_stats['attack_rate'] = (pair_stats['sandwich_attacks'] / pair_stats['total_trades'] * 100)
        pair_stats = pair_stats[pair_stats['total_trades'] >= 10]  # Filter for pairs with sufficient data
        
        print("\nToken Pairs with Highest Attack Rates (min 10 trades):")
        print(pair_stats.sort_values('attack_rate', ascending=False).head(10))
        
        # Individual token analysis
        def extract_tokens(df):
            tokens = []
            for pair in df['token_pair']:
                if pair != 'UNKNOWN':
                    try:
                        tokens.extend(str(pair).split('-'))
                    except:
                        continue
            return pd.Series(tokens).value_counts()
        
        try:
            sandwich_tokens = extract_tokens(self.sandwich_df)
            normal_tokens = extract_tokens(self.normal_df)
            
            print("\nMost Common Tokens in Sandwich Attacks:")
            print(sandwich_tokens.head(10))
            
            print("\nMost Common Tokens in Normal Trades:")
            print(normal_tokens.head(10))
        except Exception as e:
            print(f"\nWarning: Could not analyze individual tokens due to error: {str(e)}")
        
        return pair_stats
    
    def analyze_value_patterns(self):
        """Analyze patterns in transaction values"""
        print("\nTransaction Value Analysis:")
        
        # Clean USD values
        self.sandwich_df['usd_value'] = pd.to_numeric(self.sandwich_df['usd_value'], errors='coerce')
        self.normal_df['usd_value'] = pd.to_numeric(self.normal_df['usd_value'], errors='coerce')
        
        # Basic statistics
        print("\nSandwich Attack Values:")
        print(self.sandwich_df['usd_value'].describe())
        
        print("\nNormal Trade Values:")
        print(self.normal_df['usd_value'].describe())
        
        # Value range analysis
        value_ranges = [0, 100, 1000, 10000, 100000, float('inf')]
        value_labels = ['0-100', '100-1K', '1K-10K', '10K-100K', '100K+']
        
        def get_value_distribution(df):
            return pd.cut(df['usd_value'], bins=value_ranges, labels=value_labels).value_counts()
        
        sandwich_dist = get_value_distribution(self.sandwich_df)
        normal_dist = get_value_distribution(self.normal_df)
        
        print("\nValue Distribution in Sandwich Attacks:")
        for range_label, count in sandwich_dist.items():
            print(f"{range_label}: {count} attacks ({count/len(self.sandwich_df)*100:.1f}%)")
        
        return sandwich_dist, normal_dist
    
    def analyze_blockchain_patterns(self):
        """Analyze patterns across different blockchains"""
        print("\nBlockchain Analysis:")
        
        # Clean blockchain values
        self.df['blockchain'] = self.df['blockchain'].fillna('UNKNOWN')
        self.sandwich_df['blockchain'] = self.sandwich_df['blockchain'].fillna('UNKNOWN')
        
        # Attack rates by blockchain
        chain_stats = pd.DataFrame({
            'total_trades': self.df['blockchain'].value_counts(),
            'sandwich_attacks': self.sandwich_df['blockchain'].value_counts()
        }).fillna(0)
        
        chain_stats['attack_rate'] = (chain_stats['sandwich_attacks'] / chain_stats['total_trades'] * 100)
        
        print("\nAttack Rates by Blockchain:")
        print(chain_stats.sort_values('attack_rate', ascending=False))
        
        # Value patterns by blockchain
        chain_values = self.sandwich_df.groupby('blockchain')['usd_value'].agg(['mean', 'median', 'std']).fillna(0)
        print("\nValue Patterns by Blockchain (Sandwich Attacks):")
        print(chain_values)
        
        return chain_stats
    
    def analyze_project_patterns(self):
        """Analyze patterns across different projects"""
        print("\nProject Analysis:")
        
        # Clean project values
        self.df['project'] = self.df['project'].fillna('UNKNOWN')
        self.sandwich_df['project'] = self.sandwich_df['project'].fillna('UNKNOWN')
        
        # Attack rates by project
        project_stats = pd.DataFrame({
            'total_trades': self.df['project'].value_counts(),
            'sandwich_attacks': self.sandwich_df['project'].value_counts()
        }).fillna(0)
        
        project_stats['attack_rate'] = (project_stats['sandwich_attacks'] / project_stats['total_trades'] * 100)
        project_stats = project_stats[project_stats['total_trades'] >= 10]  # Filter for projects with sufficient data
        
        print("\nAttack Rates by Project (min 10 trades):")
        print(project_stats.sort_values('attack_rate', ascending=False))
        
        # Project-Token pair combinations
        project_token_stats = self.sandwich_df.groupby(['project', 'token_pair']).size()
        print("\nMost Common Project-Token Pair Combinations in Attacks:")
        print(project_token_stats.sort_values(ascending=False).head(10))
        
        return project_stats
    
    def extract_attack_rules(self, min_confidence=0.7, min_support=0.01):
        """Extract high-confidence rules for identifying sandwich attacks"""
        print("\nExtracting Attack Pattern Rules:")
        
        rules = []
        
        # Token pair rules
        pair_stats = pd.DataFrame({
            'total_trades': self.df['token_pair'].value_counts(),
            'sandwich_attacks': self.sandwich_df['token_pair'].value_counts()
        }).fillna(0)
        
        pair_stats['attack_rate'] = pair_stats['sandwich_attacks'] / pair_stats['total_trades']
        pair_stats['support'] = pair_stats['sandwich_attacks'] / len(self.sandwich_df)
        
        high_risk_pairs = pair_stats[
            (pair_stats['attack_rate'] >= min_confidence) & 
            (pair_stats['support'] >= min_support)
        ]
        
        for pair, row in high_risk_pairs.iterrows():
            rules.append({
                'type': 'token_pair',
                'condition': f"token_pair == '{pair}'",
                'confidence': row['attack_rate'],
                'support': row['support'],
                'num_attacks': row['sandwich_attacks']
            })
        
        # Project rules
        project_stats = pd.DataFrame({
            'total_trades': self.df['project'].value_counts(),
            'sandwich_attacks': self.sandwich_df['project'].value_counts()
        }).fillna(0)
        
        project_stats['attack_rate'] = project_stats['sandwich_attacks'] / project_stats['total_trades']
        project_stats['support'] = project_stats['sandwich_attacks'] / len(self.sandwich_df)
        
        high_risk_projects = project_stats[
            (project_stats['attack_rate'] >= min_confidence) & 
            (project_stats['support'] >= min_support)
        ]
        
        for project, row in high_risk_projects.iterrows():
            rules.append({
                'type': 'project',
                'condition': f"project == '{project}'",
                'confidence': row['attack_rate'],
                'support': row['support'],
                'num_attacks': row['sandwich_attacks']
            })
        
        # Value range rules
        value_ranges = [(0, 100), (100, 1000), (1000, 10000), (10000, 100000), (100000, float('inf'))]
        
        for min_val, max_val in value_ranges:
            range_attacks = len(self.sandwich_df[
                (self.sandwich_df['usd_value'] >= min_val) & 
                (self.sandwich_df['usd_value'] < max_val)
            ])
            range_total = len(self.df[
                (self.df['usd_value'] >= min_val) & 
                (self.df['usd_value'] < max_val)
            ])
            
            if range_total > 0:
                attack_rate = range_attacks / range_total
                support = range_attacks / len(self.sandwich_df)
                
                if attack_rate >= min_confidence and support >= min_support:
                    rules.append({
                        'type': 'value_range',
                        'condition': f"usd_value >= {min_val} and usd_value < {max_val}",
                        'confidence': attack_rate,
                        'support': support,
                        'num_attacks': range_attacks
                    })
        
        # Sort rules by confidence
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("\nHigh Confidence Attack Pattern Rules:")
        for rule in rules:
            print(f"\nRule Type: {rule['type']}")
            print(f"Condition: {rule['condition']}")
            print(f"Confidence: {rule['confidence']:.2%}")
            print(f"Support: {rule['support']:.2%}")
            print(f"Number of Attacks: {rule['num_attacks']}")
        
        return rules
    
    def plot_patterns(self):
        """Create visualizations of the patterns"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Value Distribution
        plt.subplot(2, 2, 1)
        plt.hist([
            np.log10(self.sandwich_df['usd_value'].fillna(0) + 1),
            np.log10(self.normal_df['usd_value'].fillna(0) + 1)
        ], label=['Sandwich', 'Normal'], bins=50, alpha=0.5)
        plt.title('Transaction Value Distribution (log scale)')
        plt.xlabel('Log10(USD Value + 1)')
        plt.ylabel('Count')
        plt.legend()
        
        # 2. Attack Rates by Blockchain
        plt.subplot(2, 2, 2)
        chain_stats = pd.DataFrame({
            'total_trades': self.df['blockchain'].value_counts(),
            'sandwich_attacks': self.sandwich_df['blockchain'].value_counts()
        }).fillna(0)
        chain_stats['attack_rate'] = chain_stats['sandwich_attacks'] / chain_stats['total_trades'] * 100
        chain_stats['attack_rate'].sort_values().plot(kind='bar')
        plt.title('Attack Rate by Blockchain')
        plt.xlabel('Blockchain')
        plt.ylabel('Attack Rate (%)')
        plt.xticks(rotation=45)
        
        # 3. Top Token Pairs in Attacks
        plt.subplot(2, 2, 3)
        self.sandwich_df['token_pair'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Token Pairs in Sandwich Attacks')
        plt.xlabel('Token Pair')
        plt.ylabel('Number of Attacks')
        plt.xticks(rotation=45)
        
        # 4. Project Attack Rates
        plt.subplot(2, 2, 4)
        project_stats = pd.DataFrame({
            'total_trades': self.df['project'].value_counts(),
            'sandwich_attacks': self.sandwich_df['project'].value_counts()
        }).fillna(0)
        project_stats['attack_rate'] = project_stats['sandwich_attacks'] / project_stats['total_trades'] * 100
        project_stats = project_stats[project_stats['total_trades'] >= 10]
        project_stats['attack_rate'].sort_values().plot(kind='bar')
        plt.title('Attack Rate by Project (min 10 trades)')
        plt.xlabel('Project')
        plt.ylabel('Attack Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('sandwich_patterns.png', dpi=300, bbox_inches='tight')
        print("\nSaved visualizations to sandwich_patterns.png")
    
    def save_analysis(self, output_file='sandwich_analysis.json'):
        """Save the complete analysis results to a JSON file"""
        analysis = {
            'summary': {
                'total_transactions': len(self.df),
                'sandwich_attacks': len(self.sandwich_df),
                'normal_trades': len(self.normal_df),
                'overall_attack_rate': len(self.sandwich_df) / len(self.df)
            },
            'token_patterns': {
                'top_pairs': self.sandwich_df['token_pair'].value_counts().head(10).to_dict(),
                'pair_attack_rates': self.analyze_token_patterns().head(10).to_dict()
            },
            'value_patterns': {
                'sandwich_stats': self.sandwich_df['usd_value'].describe().to_dict(),
                'normal_stats': self.normal_df['usd_value'].describe().to_dict()
            },
            'blockchain_patterns': self.analyze_blockchain_patterns().to_dict(),
            'project_patterns': self.analyze_project_patterns().to_dict(),
            'attack_rules': self.extract_attack_rules()
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nSaved detailed analysis to {output_file}")
        return analysis

def main():
    analyzer = SandwichPatternAnalyzer()
    
    # Load data
    analyzer.load_data('prediction_results.csv')
    
    # Run analyses
    analyzer.analyze_token_patterns()
    analyzer.analyze_value_patterns()
    analyzer.analyze_blockchain_patterns()
    analyzer.analyze_project_patterns()
    
    # Extract rules
    rules = analyzer.extract_attack_rules()
    
    # Create visualizations
    analyzer.plot_patterns()
    
    # Save complete analysis
    analyzer.save_analysis()
    
    print("\nAnalysis complete. Check the output files for detailed results.")

if __name__ == "__main__":
    main() 