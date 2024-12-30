import os
import anthropic
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import json
from bs4 import BeautifulSoup
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd

class ClaudePredictor:
    def __init__(self, api_key=None):
        """Initialize the predictor with Claude API key"""
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Please provide an API key or set ANTHROPIC_API_KEY environment variable")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.train_entries = []
        self.test_entries = []
    
    def parse_transaction(self, content):
        """Parse a transaction from content format"""
        lines = content.strip().split('\n')
        transaction = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                transaction[key.strip()] = value.strip()
        
        return transaction
    
    def load_and_split_knowledge_base(self, file_path, test_size=0.2, random_seed=42):
        """Load and split the knowledge base into training and testing sets"""
        print("Loading knowledge base...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse XML using BeautifulSoup
        soup = BeautifulSoup(content, 'xml')
        entries = []
        
        # Extract entries and track unique transaction hashes
        seen_hashes = set()
        duplicates = 0
        
        for entry in soup.find_all('entry'):
            transaction = {
                'title': entry.find('title').text if entry.find('title') else '',
                'category': entry.find('category').text if entry.find('category') else '',
                'tags': entry.find('tags').text if entry.find('tags') else '',
                'content': entry.find('content').text.strip() if entry.find('content') else ''
            }
            
            # Check for duplicate transactions and filter for USDT-WBNB pairs
            if transaction['content']:
                entry_info = self.parse_transaction(transaction['content'])
                tx_hash = entry_info.get('Transaction Hash', '').lower()
                token_pair = entry_info.get('Token Pair', '').lower()
                
                # Only include USDT-WBNB transactions (check both orders)
                if token_pair not in ['usdt-wbnb', 'wbnb-usdt']:
                    continue
                
                if tx_hash:
                    if tx_hash in seen_hashes:
                        duplicates += 1
                        continue
                    seen_hashes.add(tx_hash)
                
                entries.append(transaction)
        
        print(f"Found {len(entries)} USDT-WBNB transactions")
        print(f"Removed {duplicates} duplicate transactions")
        
        # Filter valid transactions
        valid_entries = [entry for entry in entries if entry['content']]
        print(f"Valid USDT-WBNB transactions: {len(valid_entries)}")
        
        if not valid_entries:
            raise ValueError("No valid USDT-WBNB transactions found in the knowledge base")
        
        # Separate sandwich and non-sandwich transactions
        sandwich_entries = []
        non_sandwich_entries = []
        
        for entry in valid_entries:
            entry_info = self.parse_transaction(entry['content'])
            if entry_info.get('Is Sandwich', '').lower() == 'yes':
                sandwich_entries.append(entry)
            else:
                non_sandwich_entries.append(entry)
        
        print(f"\nTotal USDT-WBNB sandwich attacks: {len(sandwich_entries)}")
        print(f"Total USDT-WBNB normal transactions: {len(non_sandwich_entries)}")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Calculate split sizes ensuring sandwich attacks in test set
        test_sandwich_size = max(int(len(sandwich_entries) * test_size), 1)  # At least 1 sandwich attack
        test_non_sandwich_size = int(len(non_sandwich_entries) * test_size)
        
        # Shuffle and split entries
        random.shuffle(sandwich_entries)
        random.shuffle(non_sandwich_entries)
        
        # Create test and train sets
        self.test_entries = sandwich_entries[:test_sandwich_size] + non_sandwich_entries[:test_non_sandwich_size]
        self.train_entries = sandwich_entries[test_sandwich_size:] + non_sandwich_entries[test_non_sandwich_size:]
        
        # Shuffle test and train sets
        random.shuffle(self.test_entries)
        random.shuffle(self.train_entries)
        
        print(f"\nSplit knowledge base:")
        print(f"Training set: {len(self.train_entries)} transactions")
        print(f"Testing set: {len(self.test_entries)} transactions")
        print(f"Testing set sandwich attacks: {sum(1 for e in self.test_entries if self.parse_transaction(e['content']).get('Is Sandwich', '').lower() == 'yes')}")
    
    def filter_relevant_entries(self, transaction_details, max_entries=100, use_test_set=False):
        """Filter entries relevant to the transaction being analyzed"""
        # Parse transaction details
        transaction_info = self.parse_transaction(transaction_details)
        transaction_hash = transaction_info.get('Transaction Hash', '').lower()
        blockchain = transaction_info.get('Blockchain', '').lower()
        project = transaction_info.get('Project', '').lower()
        token_pair = transaction_info.get('Token Pair', '').lower()
        tokens_exchanged = transaction_info.get('Tokens Exchanged', '').lower()
        usd_value = self._extract_usd_value(transaction_info.get('USD Value', '0'))
        
        # Use appropriate dataset
        entries = self.test_entries if use_test_set else self.train_entries
        
        # Filter entries based on relevance
        relevant_entries = []
        
        for entry in entries:
            entry_info = self.parse_transaction(entry['content'])
            entry_hash = entry_info.get('Transaction Hash', '').lower()
            
            # Skip if it's the same transaction
            if transaction_hash and transaction_hash == entry_hash:
                continue
            
            # Calculate relevance score
            score = 0
            
            # Exact blockchain match
            if blockchain and blockchain == entry_info.get('Blockchain', '').lower():
                score += 4
            
            # Exact project match
            if project and project == entry_info.get('Project', '').lower():
                score += 3
            
            # Token pair match (exact, partial, or token overlap)
            entry_token_pair = entry_info.get('Token Pair', '').lower()
            if token_pair and entry_token_pair:
                if token_pair == entry_token_pair:
                    score += 4  # Exact match
                else:
                    # Check token overlap
                    tokens1 = set(token_pair.split('-'))
                    tokens2 = set(entry_token_pair.split('-'))
                    overlap = len(tokens1.intersection(tokens2))
                    score += overlap * 2  # 2 points per matching token
            
            # Similar transaction value (within 50% range)
            entry_usd = self._extract_usd_value(entry_info.get('USD Value', '0'))
            if usd_value > 0 and entry_usd > 0:
                value_ratio = min(usd_value, entry_usd) / max(usd_value, entry_usd)
                if value_ratio > 0.5:  # Within 50% range
                    score += 2
            
            # Boost score for sandwich attacks in training data
            if not use_test_set and entry_info.get('Is Sandwich', '').lower() == 'yes':
                score += 2  # Give extra weight to sandwich examples
            
            # Include if score is high enough
            if score >= 4:  # Adjusted threshold
                # Remove labels from similar transactions if we're in test mode
                if use_test_set:
                    content_lines = []
                    for line in entry['content'].strip().split('\n'):
                        if not any(label in line.lower() for label in ['is sandwich:', 'is victim']):
                            content_lines.append(line)
                    relevant_entries.append((score, '\n'.join(content_lines)))
                else:
                    relevant_entries.append((score, entry['content']))
        
        # Sort by relevance score (highest first) and limit entries
        relevant_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Ensure we have a mix of sandwich and non-sandwich examples in training data
        if not use_test_set:
            sandwich_examples = []
            normal_examples = []
            for entry in relevant_entries:
                entry_info = self.parse_transaction(entry[1])
                if entry_info.get('Is Sandwich', '').lower() == 'yes':
                    sandwich_examples.append(entry[1])
                else:
                    normal_examples.append(entry[1])
            
            # Ensure at least 40% sandwich examples in training data
            max_examples = min(max_entries, len(relevant_entries))
            min_sandwich = max(2, int(max_examples * 0.4))  # At least 40% sandwich attacks
            
            # Calculate how many of each type to include
            num_sandwich = min(len(sandwich_examples), min_sandwich)
            num_normal = min(len(normal_examples), max_examples - num_sandwich)
            
            # Combine examples maintaining the ratio
            balanced_entries = (
                sandwich_examples[:num_sandwich] +
                normal_examples[:num_normal]
            )
            random.shuffle(balanced_entries)  # Shuffle to avoid pattern learning
            return '\n\n'.join(balanced_entries)
        
        return '\n\n'.join([entry[1] for entry in relevant_entries[:max_entries]])

    def _extract_usd_value(self, value_str):
        """Extract USD value from string"""
        try:
            # Remove $ and any commas, then convert to float
            cleaned = value_str.replace('$', '').replace(',', '')
            return float(cleaned)
        except:
            return 0

    def _calibrate_confidence(self, raw_confidence, prediction, evidence_strength):
        """Calibrate confidence based on evidence strength and historical performance"""
        # Base confidence from model
        confidence = raw_confidence
        
        # Adjust based on evidence strength
        if evidence_strength == 'weak':
            confidence *= 0.7  # Less penalty for weak evidence
        elif evidence_strength == 'moderate':
            confidence *= 0.85  # Less penalty for moderate evidence
        # Strong evidence keeps original confidence
        
        # Cap confidence based on prediction type
        if prediction == 'Yes':
            # Less conservative with positive predictions
            confidence = min(confidence, 90)  # Increased from 85
            # Boost confidence if evidence is strong
            if evidence_strength == 'strong':
                confidence = min(confidence * 1.1, 95)  # Boost strong positive predictions
        else:
            # More conservative with negative predictions
            confidence = min(confidence, 90)  # Reduced from 95
            if evidence_strength == 'weak':
                confidence = min(confidence, 80)  # Cap weak negative predictions lower
        
        # Ensure minimum confidence
        confidence = max(confidence, 25)  # Increased minimum confidence
        
        return round(confidence)

    def _assess_evidence_strength(self, reasoning, red_flags):
        """Assess the strength of evidence from the model's reasoning"""
        evidence_strength = 'moderate'  # Default
        
        # Count specific evidence mentions
        specific_patterns = len(re.findall(r'(pattern|similar to|matches|identical to|same as)', reasoning.lower()))
        numerical_comparisons = len(re.findall(r'\d+%|\$\d+|\d+\s*(ETH|BTC|USDT|USDC)', reasoning))
        red_flag_count = len(re.findall(r'[^.,]+[.,]', red_flags)) if red_flags else 0
        
        # Assess strength
        total_evidence = specific_patterns + numerical_comparisons + red_flag_count
        
        if total_evidence >= 5 or (specific_patterns >= 2 and numerical_comparisons >= 2):
            evidence_strength = 'strong'
        elif total_evidence <= 2 and specific_patterns == 0:
            evidence_strength = 'weak'
        
        return evidence_strength

    def analyze_training_patterns(self):
        """Analyze patterns in the training data to enhance Claude's knowledge"""
        if not self.train_entries:
            raise ValueError("Please load knowledge base first using load_and_split_knowledge_base()")
        
        # Convert training entries to DataFrame
        train_df = pd.DataFrame([
            {**self.parse_transaction(entry['content']), 'content': entry['content']}
            for entry in self.train_entries
        ])
        
        # Basic statistics
        total_transactions = len(train_df)
        sandwich_attacks = len(train_df[train_df['Is Sandwich'].str.lower() == 'yes'])
        normal_trades = total_transactions - sandwich_attacks
        
        # Calculate statistics
        stats = []
        
        # Overall statistics
        stats.append(f"Training Set Statistics:")
        stats.append(f"- Total Transactions: {total_transactions}")
        stats.append(f"- Sandwich Attacks: {sandwich_attacks} ({sandwich_attacks/total_transactions*100:.1f}%)")
        stats.append(f"- Normal Trades: {normal_trades} ({normal_trades/total_transactions*100:.1f}%)")
        
        # Project statistics
        stats.append("\nSandwich Attack Distribution by Project:")
        project_stats = train_df.groupby('Project').apply(
            lambda x: f"- {x['Project'].iloc[0]}: {len(x[x['Is Sandwich'].str.lower() == 'yes'])} attacks out of {len(x)} transactions "
            f"({len(x[x['Is Sandwich'].str.lower() == 'yes'])/len(x)*100:.1f}% attack rate)"
        )
        stats.extend(project_stats.tolist())
        
        # Token pair analysis
        stats.append("\nMost Targeted Token Pairs:")
        sandwich_pairs = train_df[train_df['Is Sandwich'].str.lower() == 'yes']['Token Pair'].value_counts().head(10)
        for pair, count in sandwich_pairs.items():
            stats.append(f"- {pair}: {count} attacks")
        
        # Token amount analysis
        def extract_token_amounts(text):
            try:
                if 'Tokens Exchanged:' in text:
                    amounts = text.split('Tokens Exchanged:')[1].split('\n')[0].strip()
                    bought, sold = amounts.split(' for ')
                    bought_amount = float(bought.split()[0])
                    sold_amount = float(sold.split()[0])
                    return bought_amount, sold_amount
            except:
                pass
            return None, None
        
        # Extract token amounts
        train_df['Bought Amount'], train_df['Sold Amount'] = zip(*train_df['content'].apply(extract_token_amounts))
        
        # Calculate statistics for sandwich attacks
        sandwich_bought = train_df[train_df['Is Sandwich'].str.lower() == 'yes']['Bought Amount'].dropna()
        sandwich_sold = train_df[train_df['Is Sandwich'].str.lower() == 'yes']['Sold Amount'].dropna()
        
        stats.append("\nToken Amount Patterns in Sandwich Attacks:")
        if not sandwich_bought.empty:
            stats.append("Bought Amounts:")
            stats.append(f"- Mean: {sandwich_bought.mean():.2f}")
            stats.append(f"- Median: {sandwich_bought.median():.2f}")
            stats.append(f"- Std Dev: {sandwich_bought.std():.2f}")
        
        if not sandwich_sold.empty:
            stats.append("\nSold Amounts:")
            stats.append(f"- Mean: {sandwich_sold.mean():.2f}")
            stats.append(f"- Median: {sandwich_sold.median():.2f}")
            stats.append(f"- Std Dev: {sandwich_sold.std():.2f}")
        
        # Transaction role analysis
        stats.append("\nTransaction Role Distribution:")
        role_counts = train_df['Transaction Role'].value_counts()
        for role, count in role_counts.items():
            stats.append(f"- {role}: {count} transactions ({count/len(train_df)*100:.1f}%)")
        
        return "\n".join(stats)

    def _convert_usd(self, value):
        """Convert USD string to float"""
        try:
            return float(str(value).replace('$', '').replace(',', ''))
        except:
            return 0.0

    def predict_sandwich_attack(self, transaction_data):
        """Predict if a transaction is part of a sandwich attack"""
        # Find similar transactions from knowledge base
        similar_transactions = self._find_similar_transactions(transaction_data)
        
        # Get relevant context from knowledge base
        knowledge_base_context = self._get_knowledge_base_context(transaction_data)
        
        # Generate prompt for Claude
        prompt = self._generate_prompt(transaction_data, similar_transactions, knowledge_base_context)
        
        # Call Claude API
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                temperature=0,
                system="You are an expert in analyzing blockchain transactions, specifically sandwich attacks in USDT-WBNB trading pairs.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Print full response
            print("\nClaude's Response:")
            print(response.content)
            
            # Parse and return prediction
            prediction = self._parse_prediction(response.content)
            return prediction
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            return None

    def _generate_prompt(self, transaction_data, similar_transactions=None, knowledge_base_context=None):
        """Generate a prompt for Claude to analyze a transaction"""
        prompt = [
            "You are an expert in analyzing blockchain transactions, specifically focusing on detecting sandwich attacks in USDT-WBNB trading pairs. "
            "A sandwich attack is a type of front-running attack where an attacker places two transactions around a victim's transaction:\n"
            "1. A buy order before the victim's transaction\n"
            "2. A sell order after the victim's transaction\n\n"
            
            "Please analyze the following transaction and determine if it is part of a sandwich attack. "
            "Consider these key factors:\n"
            "1. Block-level patterns: Look for suspicious transaction ordering and timing within the same block\n"
            "2. Token amount patterns: Compare the token amounts with typical sandwich attack patterns\n"
            "3. Address behavior: Check if the addresses involved have patterns typical of sandwich attackers\n"
            "4. Transaction roles: Consider if the transaction's role (e.g. buy/sell) fits sandwich attack patterns\n\n"
            
            "Transaction to analyze:\n"
            f"{transaction_data}\n\n"
        ]

        if knowledge_base_context:
            prompt.append(
                "Relevant knowledge base context:\n"
                f"{knowledge_base_context}\n\n"
            )

        if similar_transactions:
            prompt.append(
                "Similar transactions from the knowledge base:\n"
                f"{similar_transactions}\n\n"
            )

        prompt.append(
            "Please provide your analysis in the following format:\n"
            "Is Sandwich: [Yes/No]\n"
            "Confidence: [0-100]\n"
            "Project: [Project name]\n"
            "Token Pair: [Token pair]\n"
            "Transaction Role: [Role]\n"
            "Risk Level: [Low/Medium/High]\n"
            "Explanation: [Detailed explanation of your analysis and reasoning]\n"
            "Recommendations: [Suggestions for preventing similar attacks]"
        )

        return "\n".join(prompt)

    def _parse_prediction(self, result):
        """Parse Claude's prediction response"""
        # Extract the relevant fields from the response
        prediction = None
        confidence = None
        project = None
        token_pair = None
        role = None
        risk_level = None
        explanation = None
        recommendations = None
        
        # Get text content from response
        if isinstance(result, list) and len(result) > 0:
            text = result[0].text
        else:
            text = str(result)
        
        # Parse line by line
        current_field = None
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Is Sandwich:'):
                prediction = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                confidence = float(line.split(':', 1)[1].strip())
            elif line.startswith('Project:'):
                project = line.split(':', 1)[1].strip()
            elif line.startswith('Token Pair:'):
                token_pair = line.split(':', 1)[1].strip()
            elif line.startswith('Transaction Role:'):
                role = line.split(':', 1)[1].strip()
            elif line.startswith('Risk Level:'):
                risk_level = line.split(':', 1)[1].strip()
            elif line.startswith('Explanation:'):
                current_field = 'explanation'
                explanation = line.split(':', 1)[1].strip()
            elif line.startswith('Recommendations:'):
                current_field = 'recommendations'
                recommendations = line.split(':', 1)[1].strip()
            elif current_field == 'explanation':
                explanation += ' ' + line
            elif current_field == 'recommendations':
                recommendations += ' ' + line
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'project': project,
            'token_pair': token_pair,
            'role': role,
            'risk_level': risk_level,
            'explanation': explanation,
            'recommendations': recommendations
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=(
            retry_if_exception_type(anthropic.InternalServerError) |
            retry_if_exception_type(anthropic.APIError)
        )
    )
    def call_claude_api(self, prompt):
        """Call Claude API with retry logic specifically for overloaded errors"""
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                temperature=0,
                system="You are an expert in analyzing blockchain transactions, specifically sandwich attacks in USDT-WBNB trading pairs.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract text from the message content
            if hasattr(message.content[0], 'text'):
                return message.content[0].text
            return str(message.content[0])
            
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            raise

    def evaluate_performance(self, num_samples=5):  # Reduced default sample size
        """Evaluate model performance on test set"""
        if not self.test_entries:
            raise ValueError("No test set available. Call load_and_split_knowledge_base first.")
        
        # Separate sandwich and non-sandwich transactions in test set
        sandwich_tests = []
        non_sandwich_tests = []
        
        for entry in self.test_entries:
            entry_info = self.parse_transaction(entry['content'])
            if entry_info.get('Is Sandwich', '').lower() == 'yes':
                sandwich_tests.append(entry)
            else:
                non_sandwich_tests.append(entry)
        
        print(f"\nTest set composition:")
        print(f"Sandwich attacks: {len(sandwich_tests)}")
        print(f"Normal transactions: {len(non_sandwich_tests)}")
        
        # Ensure at least one sandwich attack in samples
        num_sandwich = min(len(sandwich_tests), max(1, int(num_samples * 0.3)))  # At least 1, up to 30% of samples
        num_non_sandwich = min(len(non_sandwich_tests), num_samples - num_sandwich)
        
        # Sample from both categories
        test_samples = (random.sample(sandwich_tests, num_sandwich) + 
                       random.sample(non_sandwich_tests, num_non_sandwich))
        random.shuffle(test_samples)
        
        print(f"\nEvaluating performance on {len(test_samples)} test samples")
        print(f"Including {num_sandwich} sandwich attacks and {num_non_sandwich} normal transactions")
        
        predictions = []
        true_labels = []
        confidences = []
        results = []
        
        for i, entry in enumerate(test_samples, 1):
            success = False
            retries = 0
            max_retries = 3
            
            while not success and retries < max_retries:
                try:
                    # Get actual label
                    entry_info = self.parse_transaction(entry['content'])
                    is_sandwich = 'Yes' if entry_info.get('Is Sandwich', '').lower() == 'yes' else 'No'
                    
                    # Get prediction
                    result = self.predict_sandwich_attack(entry['content'], use_test_set=True)
                    print(f"\nSample {i} Full Response:")
                    print("="*50)
                    print(result)
                    print("="*50)
                    
                    # Extract prediction and confidence
                    prediction = 'No'  # default
                    confidence = 0  # default
                    reasoning = ""
                    red_flags = ""
                    
                    current_section = None
                    sections = {
                        '1. Prediction:': 'prediction',
                        '2. Confidence:': 'confidence',
                        '3. Reasoning:': 'reasoning',
                        '4. Red Flags:': 'red_flags'
                    }
                    
                    for line in result.split('\n'):
                        line = line.strip()
                        for marker, section in sections.items():
                            if line.startswith(marker):
                                current_section = section
                                line = line[len(marker):].strip()
                                break
                        
                        if current_section == 'prediction':
                            prediction = 'Yes' if 'yes' in line.lower() else 'No'
                        elif current_section == 'confidence':
                            try:
                                confidence = int(''.join(filter(str.isdigit, line)))
                            except:
                                try:
                                    confidence = int(re.search(r'(\d+)%?', line).group(1))
                                except:
                                    print(f"Warning: Could not extract confidence from line: {line}")
                                    confidence = 0
                        elif current_section == 'reasoning':
                            reasoning += line + " "
                        elif current_section == 'red_flags':
                            red_flags += line + " "
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
                    true_labels.append(is_sandwich)
                    
                    # Store result details
                    results.append({
                        'transaction': entry['content'],
                        'true_label': is_sandwich,
                        'predicted_label': prediction,
                        'confidence': confidence,
                        'reasoning': reasoning.strip(),
                        'red_flags': red_flags.strip()
                    })
                    
                    print(f"Progress: {i}/{len(test_samples)} - Prediction: {prediction} (Confidence: {confidence}%)")
                    print(f"True Label: {is_sandwich}")
                    
                    success = True
                    
                except Exception as e:
                    retries += 1
                    print(f"Error on sample {i} (attempt {retries}/{max_retries}): {str(e)}")
                    if retries >= max_retries:
                        print(f"Failed to process sample {i} after {max_retries} attempts")
                        # Add default values
                        predictions.append('No')
                        confidences.append(0)
                        true_labels.append(is_sandwich)
                        results.append({
                            'transaction': entry['content'],
                            'true_label': is_sandwich,
                            'predicted_label': 'No',
                            'confidence': 0,
                            'reasoning': 'Error: Maximum retries exceeded',
                            'red_flags': ''
                        })

        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        output_file = 'prediction_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved detailed results to {output_file}")
        
        # Calculate metrics with zero_division parameter
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, pos_label='Yes', zero_division=0)
        recall = recall_score(true_labels, predictions, pos_label='Yes', zero_division=0)
        f1 = f1_score(true_labels, predictions, pos_label='Yes', zero_division=0)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print("\nDetailed Results:")
        for i, (true, pred, conf) in enumerate(zip(true_labels, predictions, confidences), 1):
            print(f"Sample {i}: True={true}, Predicted={pred}, Confidence={conf}%")
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        
        # Print confusion matrix
        from collections import Counter
        print("\nPrediction Distribution:")
        print(Counter(predictions))
        print("\nTrue Label Distribution:")
        print(Counter(true_labels))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': avg_confidence,
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels,
            'detailed_results': results
        }

    def _find_similar_transactions(self, transaction_data):
        """Find similar transactions from the knowledge base"""
        if not self.train_entries:
            return None
            
        # Parse the transaction data
        tx_info = self.parse_transaction(transaction_data)
        
        # Find transactions with same project and token pair
        similar_txs = []
        for entry in self.train_entries:
            entry_info = self.parse_transaction(entry['content'])
            if (entry_info['Project'] == tx_info['Project'] and 
                entry_info['Token Pair'] == tx_info['Token Pair']):
                similar_txs.append(entry['content'])
                
        # Return top 5 most similar transactions
        return "\n\n".join(similar_txs[:5]) if similar_txs else None
        
    def _get_knowledge_base_context(self, transaction_data):
        """Get relevant context from the knowledge base"""
        if not self.train_entries:
            return None
            
        # Parse the transaction data
        tx_info = self.parse_transaction(transaction_data)
        
        # Get statistics for this project and token pair
        train_df = pd.DataFrame([
            self.parse_transaction(entry['content'])
            for entry in self.train_entries
        ])
        
        context = []
        
        # Project stats
        project_df = train_df[train_df['Project'] == tx_info['Project']]
        project_total = len(project_df)
        if project_total > 0:
            project_attacks = len(project_df[project_df['Is Sandwich'].str.lower() == 'yes'])
            context.extend([
                f"Project Statistics for {tx_info['Project']}:",
                f"- Total transactions: {project_total}",
                f"- Sandwich attacks: {project_attacks} ({project_attacks/project_total*100:.1f}% attack rate)"
            ])
        else:
            context.extend([
                f"Project Statistics for {tx_info['Project']}:",
                "- No previous transactions found in knowledge base"
            ])
        
        # Token pair stats
        pair_df = train_df[train_df['Token Pair'] == tx_info['Token Pair']]
        pair_total = len(pair_df)
        if pair_total > 0:
            pair_attacks = len(pair_df[pair_df['Is Sandwich'].str.lower() == 'yes'])
            context.extend([
                f"\nToken Pair Statistics for {tx_info['Token Pair']}:",
                f"- Total transactions: {pair_total}",
                f"- Sandwich attacks: {pair_attacks} ({pair_attacks/pair_total*100:.1f}% attack rate)"
            ])
        else:
            context.extend([
                f"\nToken Pair Statistics for {tx_info['Token Pair']}:",
                "- No previous transactions found in knowledge base"
            ])
        
        return "\n".join(context)

def main():
    # Initialize predictor with API key
    # api_key = "empty"
    # setup the api key to for the
    # predictor = ClaudePredictor(api_key=api_key)
    
    # Load and split knowledge base
    predictor.load_and_split_knowledge_base('claude_knowledge_base.txt')
    
    # Separate sandwich and non-sandwich transactions
    sandwich_txs = []
    normal_txs = []
    for tx in predictor.test_entries:
        tx_info = predictor.parse_transaction(tx['content'])
        if tx_info.get('Is Sandwich', '').lower() == 'yes':
            sandwich_txs.append(tx)
        else:
            normal_txs.append(tx)
    
    # Ensure we have at least 10 sandwich attacks in our sample
    num_sandwich = min(len(sandwich_txs), 10)
    num_normal = 100 - num_sandwich
    
    # Get balanced sample
    test_sample = random.sample(sandwich_txs, num_sandwich) + random.sample(normal_txs, num_normal)
    random.shuffle(test_sample)
    
    print(f"\nAnalyzing {len(test_sample)} sample transactions...")
    print(f"Including {num_sandwich} sandwich attacks and {num_normal} normal transactions")
    
    results = []
    true_labels = []
    predictions = []
    
    # Create list to store detailed transaction data
    transaction_results = []
    
    for i, tx in enumerate(test_sample, 1):
        # Store true label
        tx_info = predictor.parse_transaction(tx['content'])
        true_label = 1 if tx_info.get('Is Sandwich', '').lower() == 'yes' else 0
        true_labels.append(true_label)
        
        # Remove label for prediction
        tx_lines = []
        for line in tx['content'].strip().split('\n'):
            if not line.lower().startswith('is sandwich:'):
                tx_lines.append(line)
        unlabeled_tx = '\n'.join(tx_lines)
        
        print(f"\nTransaction {i}/100:")
        print(unlabeled_tx)
        print("\nPredicting...")
        result = predictor.predict_sandwich_attack(unlabeled_tx)
        
        if result:
            pred_label = 1 if result['prediction'].lower() == 'yes' else 0
            predictions.append(pred_label)
            results.append({
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': result['confidence'],
                'risk_level': result['risk_level']
            })
            
            # Store detailed transaction data
            transaction_data = {
                'transaction_id': i,
                'blockchain': tx_info.get('Blockchain', ''),
                'project': tx_info.get('Project', ''),
                'block_number': tx_info.get('Block Number', ''),
                'block_time': tx_info.get('Block Time', ''),
                'token_pair': tx_info.get('Token Pair', ''),
                'tokens_exchanged': tx_info.get('Tokens Exchanged', ''),
                'transaction_hash': tx_info.get('Transaction Hash', ''),
                'from_address': tx_info.get('From Address', ''),
                'to_address': tx_info.get('To Address', ''),
                'event_index': tx_info.get('Event Index', ''),
                'transaction_role': tx_info.get('Transaction Role', ''),
                'true_label': 'Yes' if true_label == 1 else 'No',
                'predicted_label': result['prediction'],
                'confidence': result['confidence'],
                'risk_level': result['risk_level'],
                'explanation': result['explanation'],
                'recommendations': result['recommendations']
            }
            transaction_results.append(transaction_data)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Risk Level: {result['risk_level']}")
            print(f"True Label: {'Yes' if true_label == 1 else 'No'}")
        else:
            print("Error in prediction")
    
    # Calculate performance metrics
    if predictions and true_labels:
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        print("\nConfusion Matrix:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        # Calculate average confidence for correct and incorrect predictions
        correct_conf = [r['confidence'] for r in results if r['true_label'] == r['predicted_label']]
        incorrect_conf = [r['confidence'] for r in results if r['true_label'] != r['predicted_label']]
        
        print("\nConfidence Analysis:")
        print(f"Average confidence for correct predictions: {sum(correct_conf)/len(correct_conf):.1f}%")
        if incorrect_conf:
            print(f"Average confidence for incorrect predictions: {sum(incorrect_conf)/len(incorrect_conf):.1f}%")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results, 1):
            print(f"\nTransaction {i}:")
            print(f"True Label: {'Yes' if result['true_label'] == 1 else 'No'}")
            print(f"Predicted: {'Yes' if result['predicted_label'] == 1 else 'No'}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Risk Level: {result['risk_level']}")
        
        # Save results to CSV
        results_df = pd.DataFrame(transaction_results)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_filename = f"prediction_results_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
        
        # Save performance metrics to a separate CSV
        metrics_data = {
            'timestamp': [timestamp],
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1],
            'true_negatives': [tn],
            'false_positives': [fp],
            'false_negatives': [fn],
            'true_positives': [tp],
            'avg_correct_confidence': [sum(correct_conf)/len(correct_conf)],
            'avg_incorrect_confidence': [sum(incorrect_conf)/len(incorrect_conf)] if incorrect_conf else [0],
            'total_samples': [len(predictions)],
            'sandwich_attacks': [num_sandwich],
            'normal_transactions': [num_normal]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_filename = f"performance_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Performance metrics saved to {metrics_filename}")

if __name__ == "__main__":
    main() 