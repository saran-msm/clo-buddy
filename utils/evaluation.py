from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
from typing import Dict, Any, List, Union
import math
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

class DocumentEvaluator:
    def __init__(self):
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize GPT-2 for perplexity calculation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model.eval()

    def calculate_rouge(self, generated_text: str, reference_text: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference_text, generated_text)
            return {
                'rouge1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'fmeasure': scores['rouge1'].fmeasure
                },
                'rouge2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'fmeasure': scores['rouge2'].fmeasure
                },
                'rougeL': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'fmeasure': scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {}

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity score
        """
        try:
            tokens = word_tokenize(text.lower())
            return len(set(tokens)) / len(tokens) if tokens else 0.0
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return 0.0

    def evaluate_document(self, generated_text, reference_text):
        """
        Evaluate document using ROUGE scores and perplexity
        """
        # Compute ROUGE scores
        rouge_scores = self.calculate_rouge(generated_text, reference_text)
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(generated_text)
        
        # Generate evaluation report
        evaluation_report = {
            'rouge_scores': rouge_scores,
            'perplexity': perplexity,
            'next_steps': self.generate_next_steps(rouge_scores, perplexity)
        }
        
        return evaluation_report

    def generate_next_steps(self, rouge_scores, perplexity):
        """
        Generate actionable next steps based on evaluation metrics
        """
        next_steps = []
        
        # ROUGE score analysis
        if rouge_scores['rougeL']['fmeasure'] < 0.3:
            next_steps.append("Improve content preservation by enhancing the summarization model")
        elif rouge_scores['rougeL']['fmeasure'] < 0.5:
            next_steps.append("Fine-tune the model for better content alignment")
            
        if rouge_scores['rouge2']['fmeasure'] < 0.2:
            next_steps.append("Enhance phrase-level coherence in generated summaries")
            
        # Perplexity analysis
        if perplexity:
            if perplexity > 100:
                next_steps.append("Improve text fluency and readability")
            elif perplexity > 50:
                next_steps.append("Consider fine-tuning for better language modeling")
                
        if not next_steps:
            next_steps.append("Current performance is satisfactory. Monitor for consistency.")
            
        return next_steps

    def format_evaluation_results(self, evaluation_report):
        """
        Format evaluation results for display
        """
        formatted_results = {
            'metrics': {
                'ROUGE-1': f"P: {evaluation_report['rouge_scores']['rouge1']['precision']:.3f}, "
                          f"R: {evaluation_report['rouge_scores']['rouge1']['recall']:.3f}, "
                          f"F1: {evaluation_report['rouge_scores']['rouge1']['fmeasure']:.3f}",
                'ROUGE-2': f"P: {evaluation_report['rouge_scores']['rouge2']['precision']:.3f}, "
                          f"R: {evaluation_report['rouge_scores']['rouge2']['recall']:.3f}, "
                          f"F1: {evaluation_report['rouge_scores']['rouge2']['fmeasure']:.3f}",
                'ROUGE-L': f"P: {evaluation_report['rouge_scores']['rougeL']['precision']:.3f}, "
                          f"R: {evaluation_report['rouge_scores']['rougeL']['recall']:.3f}, "
                          f"F1: {evaluation_report['rouge_scores']['rougeL']['fmeasure']:.3f}",
                'Perplexity': f"{evaluation_report['perplexity']:.2f}" if evaluation_report['perplexity'] else "N/A"
            },
            'next_steps': evaluation_report['next_steps']
        }
        
        return formatted_results

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
        
    def evaluate(self, generated_text: str, reference_text: str) -> Dict[str, Any]:
        """
        Evaluates the generated text against the reference text
        """
        metrics = {}
        
        # Calculate ROUGE scores
        rouge_scores = self._calculate_rouge(generated_text, reference_text)
        metrics.update(rouge_scores)
        
        # Calculate Perplexity
        perplexity = self._calculate_perplexity(generated_text)
        metrics['perplexity'] = perplexity
        
        # Add recommendations based on scores
        metrics['recommendations'] = self._generate_recommendations(metrics)
        
        return metrics
    
    def _calculate_rouge(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Calculates ROUGE-L and ROUGE-Lsum scores
        """
        scores = self.scorer.score(reference_text, generated_text)
        
        return {
            'rouge_l_precision': scores['rougeL'].precision,
            'rouge_l_recall': scores['rougeL'].recall,
            'rouge_l_f1': scores['rougeL'].fmeasure,
            'rouge_lsum_f1': scores['rougeLsum'].fmeasure
        }
    
    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculates a simplified perplexity score
        Note: This is a basic implementation. For more accurate results,
        consider using a language model like GPT-2 or BERT
        """
        tokens = word_tokenize(text.lower())
        
        # Calculate token frequencies
        freq_dist = {}
        for token in tokens:
            freq_dist[token] = freq_dist.get(token, 0) + 1
        
        # Calculate probability distribution
        total_tokens = len(tokens)
        probabilities = [freq_dist[token] / total_tokens for token in tokens]
        
        # Calculate perplexity
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        perplexity = 2 ** entropy
        
        return perplexity
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> list:
        """
        Generates actionable recommendations based on evaluation metrics
        """
        recommendations = []
        
        # ROUGE-L recommendations
        if metrics['rouge_l_f1'] < 0.3:
            recommendations.append(
                "Low ROUGE-L score indicates poor content overlap. Consider improving content preservation."
            )
        
        # Perplexity recommendations
        if metrics['perplexity'] > 100:  # Threshold can be adjusted
            recommendations.append(
                "High perplexity indicates potential issues with text fluency. Consider regenerating with different parameters."
            )
        
        # Add general recommendation if scores are good
        if metrics['rouge_l_f1'] > 0.5 and metrics['perplexity'] < 50:
            recommendations.append(
                "Overall good performance. Consider fine-tuning for specific domain terminology if needed."
            )
        
        return recommendations 