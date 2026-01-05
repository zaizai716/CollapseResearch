#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of model collapse experiment results.
Generates graphs and detailed metrics report from the experiment data.
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

def load_metrics(metrics_file="nature_exact_experiment/metrics_history.json"):
    """Load metrics from the experiment"""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_file}")
        return None

def create_perplexity_plot(metrics_history):
    """Create perplexity progression plot (Nature paper primary metric)"""
    generations = []
    perplexities = []
    
    for m in metrics_history:
        gen = m['generation']
        if 'nature_metrics' in m and m['nature_metrics'].get('perplexity'):
            generations.append(gen)
            perplexities.append(m['nature_metrics']['perplexity'])
    
    if not perplexities:
        print("No perplexity data found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, perplexities, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Model Collapse: Perplexity Degradation Over Generations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add percentage increase annotation
    if len(perplexities) > 1:
        increase = ((perplexities[-1] / perplexities[0]) - 1) * 100
        plt.text(0.05, 0.95, f'Total Increase: {increase:.1f}%', 
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('graphs/perplexity_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/perplexity_progression.png")

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
            vocab_div.append(am.get('vocab_diversity', 0))
            bigram_div.append(am.get('2gram_diversity', 0))
            trigram_div.append(am.get('3gram_diversity', 0))
            fourgram_div.append(am.get('4gram_diversity', 0))
    
    if not vocab_div:
        print("No diversity metrics found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vocabulary Diversity
    axes[0, 0].plot(generations, vocab_div, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Vocabulary Diversity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Unique Words / Total Words')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2-gram Diversity
    axes[0, 1].plot(generations, bigram_div, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_title('2-gram Diversity', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Unique Bigrams / Total Bigrams')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3-gram Diversity
    axes[1, 0].plot(generations, trigram_div, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_title('3-gram Diversity', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Unique Trigrams / Total Trigrams')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4-gram Diversity
    axes[1, 1].plot(generations, fourgram_div, 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_title('4-gram Diversity', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Unique 4-grams / Total 4-grams')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('N-gram Diversity Collapse Across Generations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/diversity_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/diversity_metrics.png")

def create_probability_distribution_plot(metrics_history):
    """Create probability mass distribution plot"""
    generations = []
    top_10 = []
    top_100 = []
    top_1000 = []
    tail = []
    
    for m in metrics_history:
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            ndm = m['nature_distribution_metrics']
            if ndm.get('prob_mass_top_10'):
                generations.append(m['generation'])
                top_10.append(ndm.get('prob_mass_top_10', 0))
                top_100.append(ndm.get('prob_mass_top_100', 0))
                top_1000.append(ndm.get('prob_mass_top_1000', 0))
                tail.append(ndm.get('prob_mass_tail', 0))
    
    if not top_10:
        print("No probability distribution data found")
        return
    
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
    ax2.set_title('Loss of Tail Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Probability Mass Concentration During Model Collapse', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/probability_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/probability_distribution.png")

def create_token_diversity_plot(metrics_history):
    """Create token diversity and effective vocabulary plot"""
    generations = []
    token_div = []
    effective_vocab = []
    
    for m in metrics_history:
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            ndm = m['nature_distribution_metrics']
            if ndm.get('token_diversity') is not None:
                generations.append(m['generation'])
                token_div.append(ndm.get('token_diversity', 0))
                effective_vocab.append(ndm.get('effective_vocab_size', 0))
    
    if not token_div:
        print("No token diversity data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Token diversity
    ax1.plot(generations, token_div, 'o-', linewidth=2, markersize=8, color='teal')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Token Diversity', fontsize=12)
    ax1.set_title('Token-Level Diversity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Effective vocabulary size
    ax2.plot(generations, effective_vocab, 'o-', linewidth=2, markersize=8, color='brown')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Vocabulary Size', fontsize=12)
    ax2.set_title('Active Token Usage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage decrease annotation
    if len(effective_vocab) > 1 and effective_vocab[0] > 0:
        decrease = ((1 - effective_vocab[-1] / effective_vocab[0]) * 100)
        ax2.text(0.05, 0.95, f'Vocab Loss: {decrease:.1f}%', 
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Token-Level Collapse Indicators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/token_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/token_metrics.png")

def create_combined_collapse_plot(metrics_history):
    """Create a combined normalized plot showing all key metrics"""
    # Prepare data
    data = {
        'generation': [],
        'perplexity_norm': [],
        'vocab_diversity_norm': [],
        'token_diversity_norm': [],
        'effective_vocab_norm': []
    }
    
    # Get initial values for normalization
    initial_values = {}
    
    for m in metrics_history:
        gen = m['generation']
        data['generation'].append(gen)
        
        # Perplexity (normalized, inverted so decrease is bad)
        if 'nature_metrics' in m and m['nature_metrics'].get('perplexity'):
            ppl = m['nature_metrics']['perplexity']
            if gen == 0:
                initial_values['perplexity'] = ppl
            if 'perplexity' in initial_values:
                # Invert: lower values for higher perplexity
                data['perplexity_norm'].append(initial_values['perplexity'] / ppl)
        else:
            data['perplexity_norm'].append(None)
        
        # Vocab diversity
        if 'additional_metrics' in m:
            vd = m['additional_metrics'].get('vocab_diversity', 0)
            if gen == 0:
                initial_values['vocab_diversity'] = vd
            if 'vocab_diversity' in initial_values and initial_values['vocab_diversity'] > 0:
                data['vocab_diversity_norm'].append(vd / initial_values['vocab_diversity'])
        else:
            data['vocab_diversity_norm'].append(None)
        
        # Token diversity
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            td = m['nature_distribution_metrics'].get('token_diversity', 0)
            if gen == 0:
                initial_values['token_diversity'] = td
            if 'token_diversity' in initial_values and initial_values['token_diversity'] > 0:
                data['token_diversity_norm'].append(td / initial_values['token_diversity'])
        else:
            data['token_diversity_norm'].append(None)
        
        # Effective vocab
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            ev = m['nature_distribution_metrics'].get('effective_vocab_size', 0)
            if gen == 0:
                initial_values['effective_vocab'] = ev
            if 'effective_vocab' in initial_values and initial_values['effective_vocab'] > 0:
                data['effective_vocab_norm'].append(ev / initial_values['effective_vocab'])
        else:
            data['effective_vocab_norm'].append(None)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Plot each metric
    if any(data['perplexity_norm']):
        plt.plot([g for g, v in zip(data['generation'], data['perplexity_norm']) if v is not None],
                [v for v in data['perplexity_norm'] if v is not None],
                'o-', linewidth=2, markersize=8, label='Model Quality (1/Perplexity)')
    
    if any(data['vocab_diversity_norm']):
        plt.plot([g for g, v in zip(data['generation'], data['vocab_diversity_norm']) if v is not None],
                [v for v in data['vocab_diversity_norm'] if v is not None],
                's-', linewidth=2, markersize=8, label='Vocabulary Diversity')
    
    if any(data['token_diversity_norm']):
        plt.plot([g for g, v in zip(data['generation'], data['token_diversity_norm']) if v is not None],
                [v for v in data['token_diversity_norm'] if v is not None],
                '^-', linewidth=2, markersize=8, label='Token Diversity')
    
    if any(data['effective_vocab_norm']):
        plt.plot([g for g, v in zip(data['generation'], data['effective_vocab_norm']) if v is not None],
                [v for v in data['effective_vocab_norm'] if v is not None],
                'd-', linewidth=2, markersize=8, label='Effective Vocabulary')
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Performance')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% Degradation')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Normalized Performance (1.0 = Generation 0)', fontsize=12)
    plt.title('Model Collapse: All Metrics Combined', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig('graphs/combined_collapse.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: graphs/combined_collapse.png")

def generate_detailed_report(metrics_history):
    """Generate a detailed text report of all metrics"""
    report = []
    report.append("="*70)
    report.append("MODEL COLLAPSE EXPERIMENT - DETAILED REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-"*40)
    
    # Get first and last generations with perplexity
    first_ppl = None
    last_ppl = None
    for m in metrics_history:
        if 'nature_metrics' in m and m['nature_metrics'].get('perplexity'):
            if first_ppl is None:
                first_ppl = m['nature_metrics']['perplexity']
            last_ppl = m['nature_metrics']['perplexity']
    
    if first_ppl and last_ppl:
        report.append(f"Perplexity Increase: {first_ppl:.2f} → {last_ppl:.2f} ({(last_ppl/first_ppl-1)*100:.1f}% increase)")
    
    # Vocabulary diversity
    first_vd = None
    last_vd = None
    for m in metrics_history:
        if 'additional_metrics' in m:
            if first_vd is None:
                first_vd = m['additional_metrics'].get('vocab_diversity', 0)
            last_vd = m['additional_metrics'].get('vocab_diversity', 0)
    
    if first_vd and last_vd:
        report.append(f"Vocab Diversity Loss: {first_vd:.3f} → {last_vd:.3f} ({(1-last_vd/first_vd)*100:.1f}% loss)")
    
    report.append("")
    report.append("DETAILED METRICS BY GENERATION")
    report.append("-"*40)
    
    # Detailed metrics for each generation
    for m in metrics_history:
        report.append(f"\n### GENERATION {m['generation']} ###")
        
        # Basic metrics
        if 'nature_metrics' in m:
            nm = m['nature_metrics']
            report.append(f"Perplexity: {nm.get('perplexity', 'N/A')}")
            report.append(f"Train Loss: {nm.get('train_loss', 'N/A')}")
            report.append(f"Val Loss: {nm.get('val_loss', 'N/A')}")
        
        # Distribution metrics
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            ndm = m['nature_distribution_metrics']
            report.append("\nProbability Distribution:")
            report.append(f"  Top 10 tokens: {ndm.get('prob_mass_top_10', 0):.3f}")
            report.append(f"  Top 100 tokens: {ndm.get('prob_mass_top_100', 0):.3f}")
            report.append(f"  Top 1000 tokens: {ndm.get('prob_mass_top_1000', 0):.3f}")
            report.append(f"  Tail (>1000): {ndm.get('prob_mass_tail', 0):.3f}")
            report.append(f"  Token Diversity: {ndm.get('token_diversity', 0):.3f}")
            report.append(f"  Effective Vocab Size: {ndm.get('effective_vocab_size', 0)}")
        
        # Diversity metrics
        if 'additional_metrics' in m:
            am = m['additional_metrics']
            report.append("\nDiversity Metrics:")
            report.append(f"  Vocabulary: {am.get('vocab_diversity', 0):.3f}")
            report.append(f"  2-gram: {am.get('2gram_diversity', 0):.3f}")
            report.append(f"  3-gram: {am.get('3gram_diversity', 0):.3f}")
            report.append(f"  4-gram: {am.get('4gram_diversity', 0):.3f}")
    
    report.append("")
    report.append("="*70)
    report.append("INTERPRETATION")
    report.append("-"*40)
    
    # Interpret results
    if first_ppl and last_ppl:
        ppl_increase = (last_ppl/first_ppl - 1) * 100
        if ppl_increase > 100:
            report.append("✗ SEVERE COLLAPSE: Perplexity more than doubled")
        elif ppl_increase > 50:
            report.append("✗ SIGNIFICANT COLLAPSE: Perplexity increased by >50%")
        elif ppl_increase > 20:
            report.append("⚠ MODERATE COLLAPSE: Perplexity increased by >20%")
        else:
            report.append("✓ MILD DEGRADATION: Perplexity increase <20%")
    
    if first_vd and last_vd:
        vd_loss = (1 - last_vd/first_vd) * 100
        if vd_loss > 90:
            report.append("✗ VOCABULARY COLLAPSE: >90% diversity loss")
        elif vd_loss > 70:
            report.append("✗ SEVERE DIVERSITY LOSS: >70% reduction")
        elif vd_loss > 50:
            report.append("⚠ SIGNIFICANT DIVERSITY LOSS: >50% reduction")
        else:
            report.append("✓ MODERATE DIVERSITY LOSS: <50% reduction")
    
    report.append("")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    with open("graphs/detailed_report.txt", "w") as f:
        f.write(report_text)
    
    print("✓ Saved: graphs/detailed_report.txt")
    print("\nReport Preview:")
    print("\n".join(report[:30]))  # Show first 30 lines

def create_metrics_table(metrics_history):
    """Create a summary table of key metrics"""
    # Prepare data for pandas DataFrame
    data = []
    for m in metrics_history:
        row = {
            'Generation': m['generation'],
            'Perplexity': None,
            'Vocab Diversity': None,
            '2-gram Diversity': None,
            'Token Diversity': None,
            'Effective Vocab': None,
            'Prob Mass Top 10': None
        }
        
        if 'nature_metrics' in m:
            row['Perplexity'] = m['nature_metrics'].get('perplexity')
        
        if 'additional_metrics' in m:
            row['Vocab Diversity'] = m['additional_metrics'].get('vocab_diversity')
            row['2-gram Diversity'] = m['additional_metrics'].get('2gram_diversity')
        
        if 'nature_distribution_metrics' in m and m['nature_distribution_metrics']:
            row['Token Diversity'] = m['nature_distribution_metrics'].get('token_diversity')
            row['Effective Vocab'] = m['nature_distribution_metrics'].get('effective_vocab_size')
            row['Prob Mass Top 10'] = m['nature_distribution_metrics'].get('prob_mass_top_10')
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format the table
    pd.set_option('display.float_format', lambda x: '%.3f' % x if x is not None and x < 1 else '%.1f' % x if x is not None else 'N/A')
    
    # Save as CSV
    df.to_csv('graphs/metrics_summary.csv', index=False)
    print("✓ Saved: graphs/metrics_summary.csv")
    
    # Display
    print("\nMETRICS SUMMARY TABLE:")
    print(df.to_string(index=False))
    
    return df

def main():
    """Run all analysis and generate all outputs"""
    print("="*60)
    print("MODEL COLLAPSE EXPERIMENT ANALYSIS")
    print("="*60)
    
    # Create graphs directory
    Path("graphs").mkdir(exist_ok=True)
    
    # Load metrics
    metrics_history = load_metrics()
    if not metrics_history:
        print("ERROR: No metrics found. Run the experiment first.")
        return
    
    print(f"\nFound data for {len(metrics_history)} generations")
    print("\nGenerating visualizations...")
    
    # Generate all plots
    try:
        create_perplexity_plot(metrics_history)
    except Exception as e:
        print(f"  ✗ Perplexity plot failed: {e}")
    
    try:
        create_diversity_plots(metrics_history)
    except Exception as e:
        print(f"  ✗ Diversity plots failed: {e}")
    
    try:
        create_probability_distribution_plot(metrics_history)
    except Exception as e:
        print(f"  ✗ Probability distribution plot failed: {e}")
    
    try:
        create_token_diversity_plot(metrics_history)
    except Exception as e:
        print(f"  ✗ Token diversity plot failed: {e}")
    
    try:
        create_combined_collapse_plot(metrics_history)
    except Exception as e:
        print(f"  ✗ Combined plot failed: {e}")
    
    print("\nGenerating reports...")
    
    try:
        generate_detailed_report(metrics_history)
    except Exception as e:
        print(f"  ✗ Detailed report failed: {e}")
    
    try:
        create_metrics_table(metrics_history)
    except Exception as e:
        print(f"  ✗ Metrics table failed: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutputs generated:")
    print("  📊 graphs/perplexity_progression.png")
    print("  📊 graphs/diversity_metrics.png")
    print("  📊 graphs/probability_distribution.png")
    print("  📊 graphs/token_metrics.png")
    print("  📊 graphs/combined_collapse.png")
    print("  📄 graphs/detailed_report.txt")
    print("  📄 graphs/metrics_summary.csv")
    
    # Quick summary
    first_gen = metrics_history[0]
    last_gen = metrics_history[-1]
    
    print("\nQUICK SUMMARY:")
    if 'nature_metrics' in first_gen and 'nature_metrics' in last_gen:
        if first_gen['nature_metrics'].get('perplexity') and last_gen['nature_metrics'].get('perplexity'):
            first_ppl = first_gen['nature_metrics']['perplexity']
            last_ppl = last_gen['nature_metrics']['perplexity']
            print(f"  Perplexity: {first_ppl:.1f} → {last_ppl:.1f} ({(last_ppl/first_ppl-1)*100:+.1f}%)")
    
    if 'additional_metrics' in first_gen and 'additional_metrics' in last_gen:
        first_vd = first_gen['additional_metrics'].get('vocab_diversity', 0)
        last_vd = last_gen['additional_metrics'].get('vocab_diversity', 0)
        if first_vd > 0:
            print(f"  Vocab Diversity: {first_vd:.3f} → {last_vd:.3f} ({(last_vd/first_vd-1)*100:+.1f}%)")

if __name__ == "__main__":
    main()