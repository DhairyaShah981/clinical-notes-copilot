#!/bin/bash
# Complete workflow to evaluate RAG on medical textbooks

set -e  # Exit on error

echo "======================================================================"
echo "MEDICAL TEXTBOOK RAG EVALUATION - COMPLETE WORKFLOW"
echo "======================================================================"
echo ""

# Check if Groq API key is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ ERROR: GROQ_API_KEY not set!"
    echo ""
    echo "Get free key at: https://console.groq.com"
    echo "Then run: export GROQ_API_KEY='your_key_here'"
    exit 1
fi

echo "✓ Groq API key found"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source backend/venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Check if PDFs exist
if [ ! -f "data/anatomy_20.pdf" ]; then
    echo "❌ ERROR: data/anatomy_20.pdf not found!"
    echo "Please add your 20-page PDF to data/anatomy_20.pdf"
    exit 1
fi

if [ ! -f "data/anatomy_200.pdf" ]; then
    echo "⚠️  WARNING: data/anatomy_200.pdf not found!"
    echo "Skipping 200-page evaluation..."
    SKIP_200=true
fi

echo "======================================================================"
echo "STEP 1: Generate Q&A Pairs (Using Groq - FREE)"
echo "======================================================================"
echo ""

# Generate Q&A for 20-page book
echo "📚 Generating 10 Q&A pairs for anatomy_20.pdf..."
python generate_qa_groq.py \
    --pdf data/anatomy_20.pdf \
    --num_questions 10 \
    --output data/anatomy_20_qa.csv

echo ""
echo "✅ Generated Q&A for 20-page textbook"
echo ""

# Generate Q&A for 200-page book (if exists)
if [ "$SKIP_200" != "true" ]; then
    echo "📚 Generating 100 Q&A pairs for anatomy_200.pdf..."
    python generate_qa_groq.py \
        --pdf data/anatomy_200.pdf \
        --num_questions 100 \
        --output data/anatomy_200_qa.csv
    
    echo ""
    echo "✅ Generated Q&A for 200-page textbook"
    echo ""
fi

echo "======================================================================"
echo "STEP 2: Evaluate RAG Strategies (Hybrid vs Vector vs Keyword)"
echo "======================================================================"
echo ""

# Evaluate 20-page book
echo "🔍 Evaluating strategies on anatomy_20.pdf..."
echo "   This will test: Hybrid, Vector-Only, Keyword-Only"
echo "   Time: ~5-10 minutes"
echo ""

python evaluate_textbook.py \
    --pdf data/anatomy_20.pdf \
    --qa data/anatomy_20_qa.csv \
    --output results_anatomy_20

echo ""
echo "✅ Completed evaluation for 20-page textbook"
echo ""

# Evaluate 200-page book (if exists)
if [ "$SKIP_200" != "true" ]; then
    echo "🔍 Evaluating strategies on anatomy_200.pdf..."
    echo "   This will test: Hybrid, Vector-Only, Keyword-Only"
    echo "   Time: ~30-60 minutes"
    echo ""
    
    python evaluate_textbook.py \
        --pdf data/anatomy_200.pdf \
        --qa data/anatomy_200_qa.csv \
        --output results_anatomy_200
    
    echo ""
    echo "✅ Completed evaluation for 200-page textbook"
    echo ""
fi

echo "======================================================================"
echo "✅ EVALUATION COMPLETE!"
echo "======================================================================"
echo ""
echo "📊 Results saved to:"
echo "   - results_anatomy_20/comparison_report.md"
if [ "$SKIP_200" != "true" ]; then
    echo "   - results_anatomy_200/comparison_report.md"
fi
echo ""
echo "📈 View comparison:"
echo "   cat results_anatomy_20/comparison_report.md"
echo ""
echo "======================================================================"

