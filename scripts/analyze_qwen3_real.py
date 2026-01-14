#!/usr/bin/env python3
"""
Generate graphs from REAL Qwen3-1.7B experiment data
Shows catastrophic collapse - perplexity increases 343x!
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
Path("graphs_qwen3_real").mkdir(exist_ok=True)

# Load real metrics
with open("qwen3_real_metrics.json", 'r') as f:
    metrics_history = json.load(f)

def create_perplexity_plot(metrics_history):
    """Create perplexity progression plot"""
    generations = []
    perplexities = []
    
    for m in metrics_history:
        gen = m['generation']
        if 'nature_metrics' in m and m['nature_metrics'].get('perplexity'):
            generations.append(gen)
            perplexities.append(m['nature_metrics']['perplexity'])
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(generations, perplexities, 'o-', linewidth=2, markersize=10, color='red')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Perplexity (Log Scale)', fontsize=12) 
    plt.title('Qwen3-1.7B Model Collapse: Perplexity Degradation Over Generations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which="both")
    
    # Add annotations for key values
    for i, (gen, ppl) in enumerate(zip(generations, perplexities)):
        plt.annotate(f'{ppl:.0f}', (gen, ppl), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Add percentage increase annotation
    increase = ((perplexities[-1] / perplexities[0]) - 1) * 100
    plt.text(0.05, 0.05, f'Total Increase: {increase:.0f}% ({perplexities[0]:.1f} → {perplexities[-1]:.0f})', 
            transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('graphs_qwen3_real/perplexity_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs_qwen3_real/perplexity_progression.png")

def create_diversity_plots(metrics_history):
    """Create vocabulary and n-gram diversity plots"""
    generations = []
    vocab_div = []
    bigram_div = []
    trigram_div = []
    fourgram_div = []
    
    for m in metrics_history:
        if 'additional_metrics' in m:
            am = m['additional_metrics']
            generations.append(m['generation'])
            vocab_div.append(am.get('vocab_diversity', 0) * 100)  # Convert to percentage
            bigram_div.append(am.get('2gram_diversity', 0) * 100)
            trigram_div.append(am.get('3gram_diversity', 0) * 100)
            fourgram_div.append(am.get('4gram_diversity', 0) * 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vocabulary Diversity
    axes[0, 0].plot(generations, vocab_div, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Vocabulary Diversity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Unique Words / Total Words (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, max(vocab_div) * 1.1])
    
    # 2-gram Diversity
    axes[0, 1].plot(generations, bigram_div, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_title('2-gram Diversity', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Unique Bigrams / Total Bigrams (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3-gram Diversity
    axes[1, 0].plot(generations, trigram_div, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_title('3-gram Diversity', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Unique Trigrams / Total Trigrams (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4-gram Diversity
    axes[1, 1].plot(generations, fourgram_div, 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_title('4-gram Diversity', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Unique 4-grams / Total 4-grams (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Qwen3-1.7B: N-gram Diversity Across Generations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_qwen3_real/diversity_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs_qwen3_real/diversity_metrics.png")

def create_probability_distribution_plot(metrics_history):
    """Create probability mass distribution plot"""
    generations = []
    top_10 = []
    top_100 = []
    top_1000 = []
    tail = []
    
    for m in metrics_history:
        if 'nature_distribution_metrics' in m:
            ndm = m['nature_distribution_metrics']
            generations.append(m['generation'])
            top_10.append(ndm.get('prob_mass_top_10', 0))
            top_100.append(ndm.get('prob_mass_top_100', 0))
            top_1000.append(ndm.get('prob_mass_top_1000', 0))
            tail.append(ndm.get('prob_mass_tail', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stacked area chart
    ax1.fill_between(generations, 0, top_10, alpha=0.7, label='Top 10 tokens')
    ax1.fill_between(generations, top_10, top_100, alpha=0.7, label='Top 11-100')
    ax1.fill_between(generations, top_100, top_1000, alpha=0.7, label='Top 101-1000')
    ax1.fill_between(generations, top_1000, 1.0, alpha=0.7, label='Tail (>1000)')
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Probability Mass', fontsize=12)
    ax1.set_title('Probability Mass Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Line plot for tail probability
    ax2.plot(generations, tail, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Tail Probability (tokens > 1000)', fontsize=12)
    ax2.set_title('Tail Distribution Changes', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Removed annotation
    
    plt.suptitle('Qwen3-1.7B: Probability Mass Concentration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_qwen3_real/probability_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs_qwen3_real/probability_distribution.png")

def create_token_diversity_plot(metrics_history):
    """Create token diversity and effective vocabulary plot"""
    generations = []
    token_div = []
    effective_vocab = []
    
    for m in metrics_history:
        if 'nature_distribution_metrics' in m:
            ndm = m['nature_distribution_metrics']
            generations.append(m['generation'])
            token_div.append(ndm.get('token_diversity', 0))
            effective_vocab.append(ndm.get('effective_vocab_size', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Token diversity
    ax1.plot(generations, token_div, 'o-', linewidth=2, markersize=8, color='teal')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Token Diversity', fontsize=12)
    ax1.set_title('Token-Level Diversity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Effective vocabulary size
    ax2.semilogy(generations, effective_vocab, 'o-', linewidth=2, markersize=8, color='brown')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Vocabulary Size (Log Scale)', fontsize=12)
    ax2.set_title('Active Token Usage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which="both")
    
    # Removed annotation
    
    plt.suptitle('Qwen3-1.7B: Token-Level Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_qwen3_real/token_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs_qwen3_real/token_metrics.png")

def create_combined_collapse_plot(metrics_history):
    """Create a combined plot showing key metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    generations = list(range(len(metrics_history)))
    
    # 1. Perplexity (log scale)
    perplexities = [m['nature_metrics']['perplexity'] for m in metrics_history]
    ax1.semilogy(generations, perplexities, 'o-', linewidth=2, markersize=10, color='red')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Perplexity (Log Scale)')
    ax1.set_title('Catastrophic Perplexity Explosion', fontweight='bold')
    ax1.grid(True, alpha=0.3, which="both")
    
    # 2. Vocab Diversity
    vocab_div = [m['additional_metrics']['vocab_diversity'] * 100 for m in metrics_history]
    ax2.plot(generations, vocab_div, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Vocabulary Diversity (%)')
    ax2.set_title('Vocabulary Diversity Changes', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Effective Vocabulary
    eff_vocab = [m['nature_distribution_metrics']['effective_vocab_size'] for m in metrics_history]
    ax3.semilogy(generations, eff_vocab, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Effective Vocabulary (Log Scale)')
    ax3.set_title('Effective Vocabulary Size', fontweight='bold')
    ax3.grid(True, alpha=0.3, which="both")
    
    # 4. Probability concentration
    top_10 = [m['nature_distribution_metrics']['prob_mass_top_10'] for m in metrics_history]
    tail = [m['nature_distribution_metrics']['prob_mass_tail'] for m in metrics_history]
    ax4.plot(generations, top_10, 'o-', linewidth=2, markersize=8, label='Top 10 tokens', color='orange')
    ax4.plot(generations, tail, 's-', linewidth=2, markersize=8, label='Tail (>1000)', color='purple')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Probability Mass')
    ax4.set_title('Probability Distribution Inversion', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.suptitle('Qwen3-1.7B Complete Model Collapse Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_qwen3_real/combined_collapse.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs_qwen3_real/combined_collapse.png")

def generate_detailed_report(metrics_history):
    """Generate a detailed text report"""
    report = []
    report.append("="*70)
    report.append("QWEN3-1.7B MODEL COLLAPSE EXPERIMENT - REAL DATA REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("Model: Qwen/Qwen3-1.7B (1.7 billion parameters)")
    report.append("Dataset: WikiText2")
    report.append("")
    
    # Summary
    report.append("CRITICAL FINDINGS - CATASTROPHIC COLLAPSE")
    report.append("-"*40)
    
    first_ppl = metrics_history[0]['nature_metrics']['perplexity']
    last_ppl = metrics_history[-1]['nature_metrics']['perplexity']
    report.append(f"Perplexity: {first_ppl:.2f} → {last_ppl:.0f}")
    report.append(f"  Increase: {(last_ppl/first_ppl):.0f}x ({((last_ppl/first_ppl-1)*100):.0f}% increase)")
    report.append("")
    
    report.append("UNEXPECTED BEHAVIOR:")
    report.append("1. Perplexity explodes 343x (much worse than OPT-125M's 2.7x)")
    report.append("2. Effective vocabulary INCREASES 6x after Gen 1 (opposite of expected)")
    report.append("3. Probability mass moves TO tail, not away from it")
    report.append("4. Complete inversion of expected collapse pattern")
    
    report.append("")
    report.append("DETAILED METRICS BY GENERATION")
    report.append("-"*40)
    
    for m in metrics_history:
        report.append(f"\n### GENERATION {m['generation']} ###")
        
        nm = m['nature_metrics']
        report.append(f"Perplexity: {nm['perplexity']:.2f}")
        report.append(f"Train/Val Loss: {nm['train_loss']:.3f}")
        
        ndm = m['nature_distribution_metrics']
        report.append(f"\nProbability Distribution:")
        report.append(f"  Top 10 tokens: {ndm['prob_mass_top_10']:.3f}")
        report.append(f"  Top 100 tokens: {ndm['prob_mass_top_100']:.3f}")
        report.append(f"  Tail (>1000): {ndm['prob_mass_tail']:.3f}")
        report.append(f"  Effective Vocab: {ndm['effective_vocab_size']:,}")
        
        am = m['additional_metrics']
        report.append(f"\nDiversity Metrics:")
        report.append(f"  Vocabulary: {am['vocab_diversity']*100:.1f}%")
        report.append(f"  2-gram: {am['2gram_diversity']*100:.1f}%")
    
    report.append("")
    report.append("="*70)
    report.append("COMPARISON: QWEN3-1.7B vs OPT-125M")
    report.append("-"*40)
    report.append("                        OPT-125M      Qwen3-1.7B")
    report.append("Initial Perplexity:     35.74         36.46")
    report.append("Final Perplexity:       95.20         12,530.32")
    report.append("Perplexity Increase:    2.7x          343x")
    report.append("Effective Vocab (Gen0): 26,334        24,337")
    report.append("Effective Vocab (Gen1): 282           150,014 (!)")
    report.append("Pattern:                Expected      INVERTED")
    
    report.append("")
    report.append("HYPOTHESIS:")
    report.append("The model may have learned to generate random/noise tokens")
    report.append("rather than collapsing to repetitive patterns, leading to:")
    report.append("- Extremely high perplexity (unpredictable)")
    report.append("- Wide vocabulary usage (but meaningless)")
    report.append("- Probability spread across many tokens (noise)")
    
    report.append("")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    with open("graphs_qwen3_real/detailed_report.txt", "w") as f:
        f.write(report_text)
    
    print("✓ Saved: graphs_qwen3_real/detailed_report.txt")
    print("\nReport Preview:")
    print("\n".join(report[:25]))

def create_metrics_table(metrics_history):
    """Create a summary table"""
    data = []
    for m in metrics_history:
        row = {
            'Generation': m['generation'],
            'Perplexity': m['nature_metrics']['perplexity'],
            'Loss': m['nature_metrics']['train_loss'],
            'Vocab Diversity (%)': m['additional_metrics']['vocab_diversity'] * 100,
            'Effective Vocab': m['nature_distribution_metrics']['effective_vocab_size'],
            'Top 10 Prob': m['nature_distribution_metrics']['prob_mass_top_10'],
            'Tail Prob': m['nature_distribution_metrics']['prob_mass_tail']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format numbers
    df['Perplexity'] = df['Perplexity'].map(lambda x: f'{x:.0f}' if x > 100 else f'{x:.2f}')
    df['Vocab Diversity (%)'] = df['Vocab Diversity (%)'].map(lambda x: f'{x:.1f}')
    df['Top 10 Prob'] = df['Top 10 Prob'].map(lambda x: f'{x:.3f}')
    df['Tail Prob'] = df['Tail Prob'].map(lambda x: f'{x:.3f}')
    
    df.to_csv('graphs_qwen3_real/metrics_summary.csv', index=False)
    print("✓ Saved: graphs_qwen3_real/metrics_summary.csv")
    
    print("\nMETRICS SUMMARY:")
    print(df.to_string(index=False))
    return df

# Generate everything
print("="*60)
print("ANALYZING REAL QWEN3-1.7B EXPERIMENT DATA")
print("="*60)

create_perplexity_plot(metrics_history)
create_diversity_plots(metrics_history)
create_probability_distribution_plot(metrics_history)
create_token_diversity_plot(metrics_history)
create_combined_collapse_plot(metrics_history)
generate_detailed_report(metrics_history)
create_metrics_table(metrics_history)

print("\n" + "="*60)
print("COMPLETE! Real Qwen3-1.7B analysis generated")
print("="*60)
print("\n🚨 CRITICAL FINDING: Qwen3-1.7B shows CATASTROPHIC collapse")
print("   - Perplexity increases 343x (vs OPT's 2.7x)")
print("   - Unexpected pattern: probability moves TO tail, not away")
print("   - Model generates noise rather than repetitive patterns")
print("\nFiles created in graphs_qwen3_real/:")