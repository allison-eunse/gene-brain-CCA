#!/usr/bin/env python3
"""
Generate PDF reports from markdown files with professional styling.
Uses fpdf (1.7.2) for PDF generation.
"""

import sys
sys.path.insert(0, '/home/connectome/allie/.local/lib/python3.7/site-packages')

from fpdf import FPDF
from pathlib import Path
import re
import os

# Base directory
BASE_DIR = Path("/storage/bigdata/UKB/fMRI/gene-brain-CCA/final_report")


class MarkdownPDF(FPDF):
    """Custom PDF class for rendering markdown-like content."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        
        # Try to add Unicode-supporting fonts
        fonts_added = False
        dejavu_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
            '/usr/share/fonts/dejavu/DejaVuSerif.ttf',
        ]
        
        for path in dejavu_paths:
            if os.path.exists(path):
                try:
                    base = os.path.dirname(path)
                    self.add_font('DejaVu', '', os.path.join(base, 'DejaVuSerif.ttf'), uni=True)
                    self.add_font('DejaVu', 'B', os.path.join(base, 'DejaVuSerif-Bold.ttf'), uni=True)
                    self.add_font('DejaVu', 'I', os.path.join(base, 'DejaVuSerif-Italic.ttf'), uni=True)
                    self.add_font('DejaVuMono', '', os.path.join(base, 'DejaVuSansMono.ttf'), uni=True)
                    fonts_added = True
                    break
                except Exception as e:
                    print(f"Warning: Could not add DejaVu fonts: {e}")
        
        if not fonts_added:
            print("Using built-in fonts (some characters may not display correctly)")
            self.main_font = 'Times'
        else:
            self.main_font = 'DejaVu'
        
        self.title_text = ''
        
    def header(self):
        if self.page_no() > 1:
            self.set_font(self.main_font, 'I' if self.main_font == 'Times' else '', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, self.title_text if self.title_text else 'Gene-Brain CCA Analysis', 0, 0, 'C')
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(20, 18, 190, 18)
            self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font(self.main_font, '', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Page %d' % self.page_no(), 0, 0, 'C')


def parse_markdown_content(md_content):
    """Parse markdown content into structured elements."""
    lines = md_content.split('\n')
    elements = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Headers
        if line.startswith('# '):
            elements.append(('h1', line[2:].strip()))
        elif line.startswith('## '):
            elements.append(('h2', line[3:].strip()))
        elif line.startswith('### '):
            elements.append(('h3', line[4:].strip()))
        elif line.startswith('#### '):
            elements.append(('h4', line[5:].strip()))
        
        # Horizontal rule
        elif line.strip() == '---':
            elements.append(('hr', ''))
        
        # Table
        elif '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            i -= 1  # Back up one
            elements.append(('table', table_lines))
        
        # Code block
        elif line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            elements.append(('code', '\n'.join(code_lines)))
        
        # List item
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            elements.append(('li', line.strip()[2:]))
        elif re.match(r'^\d+\.', line.strip()):
            elements.append(('li', re.sub(r'^\d+\.\s*', '', line.strip())))
        
        # Empty line
        elif line.strip() == '':
            elements.append(('space', ''))
        
        # Regular paragraph
        else:
            # Collect continuous paragraph lines
            para_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not any([
                lines[j].startswith('#'),
                lines[j].strip().startswith('- '),
                lines[j].strip().startswith('* '),
                lines[j].strip() == '---',
                lines[j].strip().startswith('```'),
                re.match(r'^\d+\.', lines[j].strip())
            ]):
                # Check for table
                if '|' in lines[j] and j + 1 < len(lines) and '---' in lines[j + 1]:
                    break
                para_lines.append(lines[j])
                j += 1
            i = j - 1
            elements.append(('p', ' '.join(para_lines)))
        
        i += 1
    
    return elements


def clean_text(text):
    """Clean markdown formatting from text and handle special characters."""
    # Remove bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove italic markers
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove anchor tags
    text = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Replace special unicode characters that might not render
    text = text.replace('→', '->')
    text = text.replace('↔', '<->')
    text = text.replace('≥', '>=')
    text = text.replace('≤', '<=')
    text = text.replace('≈', '~')
    text = text.replace('×', 'x')
    text = text.replace('ρ', 'rho')
    text = text.replace('Δ', 'Delta')
    return text


def safe_text(text, max_len=None):
    """Make text safe for PDF output."""
    text = clean_text(text)
    # Replace any remaining problematic characters
    try:
        text.encode('latin-1')
    except UnicodeEncodeError:
        # Replace non-latin-1 characters
        new_text = []
        for char in text:
            try:
                char.encode('latin-1')
                new_text.append(char)
            except UnicodeEncodeError:
                # Check if it's Korean
                if '\uac00' <= char <= '\ud7af':
                    new_text.append(char)  # Keep Korean
                else:
                    new_text.append('?')
        text = ''.join(new_text)
    
    if max_len:
        text = text[:max_len]
    return text


def render_comprehensive_pdf():
    """Generate the comprehensive PDF report."""
    print("Generating comprehensive PDF report...")
    
    # Read markdown
    md_path = BASE_DIR / "comprehensive_report.md"
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse content
    elements = parse_markdown_content(md_content)
    
    # Create PDF
    pdf = MarkdownPDF()
    pdf.title_text = 'Gene-Brain CCA Analysis: Comprehensive Report'
    pdf.set_title(pdf.title_text)
    pdf.set_author('Allie')
    pdf.add_page()
    
    main_font = pdf.main_font
    
    for elem_type, content in elements:
        try:
            if elem_type == 'h1':
                content = safe_text(content)
                pdf.ln(5)
                pdf.set_font(main_font, 'B', 18)
                pdf.set_text_color(13, 59, 102)  # Dark blue
                pdf.multi_cell(0, 10, content)
                pdf.set_draw_color(13, 59, 102)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(5)
            
            elif elem_type == 'h2':
                content = safe_text(content)
                pdf.ln(8)
                pdf.set_font(main_font, 'B', 14)
                pdf.set_text_color(26, 82, 118)  # Medium blue
                pdf.multi_cell(0, 8, content)
                pdf.set_draw_color(26, 82, 118)
                pdf.line(20, pdf.get_y(), 140, pdf.get_y())
                pdf.ln(3)
            
            elif elem_type == 'h3':
                content = safe_text(content)
                pdf.ln(5)
                pdf.set_font(main_font, 'B', 12)
                pdf.set_text_color(36, 113, 163)  # Light blue
                pdf.multi_cell(0, 7, content)
                pdf.ln(2)
            
            elif elem_type == 'h4':
                content = safe_text(content)
                pdf.ln(3)
                pdf.set_font(main_font, 'B', 11)
                pdf.set_text_color(52, 73, 94)  # Gray
                pdf.multi_cell(0, 6, content)
                pdf.ln(1)
            
            elif elem_type == 'p':
                content = safe_text(content)
                if content.strip():
                    pdf.set_font(main_font, '', 10)
                    pdf.set_text_color(0, 0, 0)
                    pdf.multi_cell(0, 5, content)
                    pdf.ln(2)
            
            elif elem_type == 'li':
                content = safe_text(content)
                pdf.set_font(main_font, '', 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(10, 5, '  *')
                pdf.multi_cell(0, 5, content)
            
            elif elem_type == 'code':
                pdf.ln(2)
                pdf.set_fill_color(44, 62, 80)  # Dark gray
                pdf.set_text_color(236, 240, 241)  # Light gray
                if 'DejaVuMono' in [f for f in pdf.fonts]:
                    pdf.set_font('DejaVuMono', '', 8)
                else:
                    pdf.set_font('Courier', '', 8)
                lines = content.split('\n')
                for line in lines:
                    line = safe_text(line)
                    pdf.cell(0, 4, '  ' + line, 0, 1, fill=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            
            elif elem_type == 'hr':
                pdf.ln(5)
                pdf.set_draw_color(189, 195, 199)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(5)
            
            elif elem_type == 'table':
                pdf.ln(3)
                table_lines = content
                # Parse table
                rows = []
                for tl in table_lines:
                    if '---' not in tl:
                        cells = [c.strip() for c in tl.split('|')[1:-1]]
                        rows.append(cells)
                
                if rows:
                    # Calculate column widths
                    num_cols = len(rows[0])
                    col_width = min(170 / num_cols, 60)
                    
                    # Header
                    pdf.set_font(main_font, 'B', 9)
                    pdf.set_fill_color(13, 59, 102)
                    pdf.set_text_color(255, 255, 255)
                    for cell in rows[0]:
                        cell = safe_text(cell, 30)
                        pdf.cell(col_width, 7, cell, 1, 0, 'L', fill=True)
                    pdf.ln()
                    
                    # Data rows
                    pdf.set_font(main_font, '', 9)
                    pdf.set_text_color(0, 0, 0)
                    for i, row in enumerate(rows[1:]):
                        if i % 2 == 0:
                            pdf.set_fill_color(248, 249, 250)
                        else:
                            pdf.set_fill_color(255, 255, 255)
                        for cell in row:
                            cell = safe_text(cell, 35)
                            pdf.cell(col_width, 6, cell, 1, 0, 'L', fill=True)
                        pdf.ln()
                pdf.ln(3)
            
            elif elem_type == 'space':
                pdf.ln(2)
        except Exception as e:
            print(f"Warning: Error processing element {elem_type}: {e}")
            continue
    
    # Save PDF
    pdf_path = BASE_DIR / "Gene-Brain_CCA_Comprehensive_Report.pdf"
    pdf.output(str(pdf_path), 'F')
    print(f"  Comprehensive PDF saved to: {pdf_path}")
    return pdf_path


def create_concise_bilingual_content():
    """Create the concise bilingual markdown content."""
    return """# Gene-Brain CCA Analysis: Concise Report

**Author:** Allie
**Date:** January 14, 2026
**Dataset:** UK Biobank (N=4,218)

---

## Executive Summary

This study investigated whether combining genetic embeddings (from DNABERT-2 foundation model) with brain imaging (fMRI) data improves Major Depressive Disorder (MDD) prediction.

**Key Findings:**
- Gene-only prediction achieves AUC 0.759 (holdout)
- fMRI adds minimal/no predictive value (early fusion AUC 0.762, +0.003)
- Unsupervised CCA/SCCA underperforms direct supervised learning by 17-23 AUC points
- Full 768-D embeddings improve performance by +29% vs scalar pooling

---

## Korean Summary (한국어 요약)

본 연구는 DNABERT-2 foundation model의 genetic embeddings과 brain imaging (fMRI) 데이터를 결합하여 Major Depressive Disorder (MDD) 예측을 향상시킬 수 있는지 조사하였습니다.

주요 결과:
- Gene-only 예측은 AUC 0.759 달성 (holdout)
- fMRI는 예측 가치를 추가하지 않음 (early fusion AUC 0.762, +0.003)
- Unsupervised CCA/SCCA는 direct supervised learning보다 17-23 AUC points 낮은 성능
- Full 768-D embeddings은 scalar pooling 대비 +29% 성능 향상

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total subjects | 4,218 |
| MDD cases | 1,735 (41.1%) |
| Controls | 2,483 (58.9%) |
| Gene features | 111 genes x 768-D |
| fMRI features | 180 brain ROIs |

---

## Methods Summary

### Experiment 1: Two-Stage CCA/SCCA

**Stage 1 (Unsupervised):** CCA/SCCA finds gene-brain correlations
**Stage 2 (Supervised):** Predict MDD from canonical variates

**Gene reduction strategies:**
- Mean pooling: Average of 768 dimensions
- Max pooling: Maximum of 768 dimensions

### Experiment 2: Leakage-Safe Pipelines

**Pipeline A:** Interpretable SCCA on scalar genes
**Pipeline B:** Supervised prediction with full 768-D embeddings

---

## Key Results

### Experiment 1: Mean vs Max Pooling

| Metric | Mean Pooling | Max Pooling |
|--------|--------------|-------------|
| Stage 1 p-value | 0.040 (sig) | 0.995 (n.s.) |
| Gene-only AUC | 0.588 | 0.505 |

Mean pooling preserves more predictive information.
Max pooling destroys the genetic signal.

### Experiment 2: Pipeline B Results

| Model | Holdout AUC | Note |
|-------|-------------|------|
| gene_only_logreg | 0.759 | Best |
| early_fusion_logreg | 0.762 | Marginal |
| fmri_only_logreg | 0.559 | Chance |
| cca_joint_logreg | 0.546 | Weak |
| scca_joint_logreg | 0.566 | Poor |

---

## Master Comparison

| Experiment | Best AUC | Key Insight |
|------------|----------|-------------|
| Exp1 Mean Pool | 0.588 | Mean preserves signal |
| Exp1 Max Pool | 0.522 | Max loses signal |
| Exp2 Pipeline B | 0.762 | Full embeddings best |
| Yoon et al. | 0.851 | Reference (N=29k) |

---

## Scientific Conclusions

| Finding | Evidence |
|---------|----------|
| Gene-brain coupling weak | r=0.368, p=0.04 |
| CCA/SCCA hurts prediction | 0.566 vs 0.759 AUC |
| fMRI adds minimal/no value | AUC 0.50-0.56 |
| Full embeddings essential | +29% improvement |

---

## Clinical Implications

### English
1. Brain imaging does not improve genetic prediction of MDD
2. Foundation model embeddings must be preserved (not pooled)
3. Gene-brain alignment is statistically real but clinically irrelevant

### Korean (한국어)
1. Brain imaging은 MDD의 genetic 예측을 향상시키지 않음
2. Foundation model embeddings은 보존되어야 함
3. Gene-brain alignment은 통계적으로 실재하나 임상적으로 무관

---

## Recommendations

### Immediate Next Steps
1. Gene curation - Filter to Yoon's 38 genes (Expected AUC: 0.80-0.84)
2. Remove PCA bottleneck - Use LASSO on full 85K features (Expected AUC: 0.78-0.82)
3. Match methodology - Implement 10-fold nested CV

### Future Directions
- Expand sample size (target N>10,000)
- Test alternative brain features (network-specific)
- Explore fMRI foundation models (BrainLM)
- Supervised feature selection for interpretability

### 향후 방향 (Korean)
- Sample size 확장 (N>10,000 목표)
- 대체 brain features 테스트
- fMRI foundation models 탐색 (BrainLM)
- 해석 가능성을 위한 supervised feature selection

---

## Technical Glossary

| Term | Definition | Korean |
|------|------------|--------|
| AUC | Area Under ROC Curve | ROC 곡선 아래 면적 |
| CCA | Canonical Correlation Analysis | 정준상관분석 |
| SCCA | Sparse CCA | 희소 정준상관분석 |
| Foundation Model | Pre-trained neural network | 대규모 사전 훈련 신경망 |
| fMRI | Functional MRI | 기능적 자기공명영상 |
| PCA | Principal Component Analysis | 주성분 분석 |
| Holdout Set | Fixed test set | 고정 테스트 셋 |
| MDD | Major Depressive Disorder | 주요우울장애 |

---

**End of Report / 보고서 끝**
"""


def render_concise_pdf():
    """Generate the concise bilingual PDF report."""
    print("Generating concise bilingual PDF report...")
    
    # Get content
    md_content = create_concise_bilingual_content()
    
    # Save markdown
    md_path = BASE_DIR / "concise_report_bilingual.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  Concise markdown saved to: {md_path}")
    
    # Parse content
    elements = parse_markdown_content(md_content)
    
    # Create PDF
    pdf = MarkdownPDF()
    pdf.title_text = 'Gene-Brain CCA Analysis: Concise Report'
    pdf.set_title(pdf.title_text)
    pdf.set_author('Allie')
    pdf.add_page()
    
    main_font = pdf.main_font
    
    for elem_type, content in elements:
        try:
            if elem_type == 'h1':
                content = safe_text(content)
                pdf.ln(5)
                pdf.set_font(main_font, 'B', 16)
                pdf.set_text_color(30, 58, 95)
                pdf.multi_cell(0, 9, content)
                pdf.set_draw_color(30, 58, 95)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(4)
            
            elif elem_type == 'h2':
                content = safe_text(content)
                pdf.ln(6)
                pdf.set_font(main_font, 'B', 13)
                pdf.set_text_color(44, 82, 130)
                pdf.multi_cell(0, 7, content)
                pdf.set_draw_color(44, 82, 130)
                pdf.line(20, pdf.get_y(), 120, pdf.get_y())
                pdf.ln(2)
            
            elif elem_type == 'h3':
                content = safe_text(content)
                pdf.ln(4)
                pdf.set_font(main_font, 'B', 11)
                pdf.set_text_color(49, 130, 206)
                pdf.multi_cell(0, 6, content)
                pdf.ln(1)
            
            elif elem_type == 'h4':
                content = safe_text(content)
                pdf.ln(2)
                pdf.set_font(main_font, 'B', 10)
                pdf.set_text_color(74, 85, 104)
                pdf.multi_cell(0, 5, content)
            
            elif elem_type == 'p':
                content = safe_text(content)
                if content.strip():
                    pdf.set_font(main_font, '', 9)
                    pdf.set_text_color(0, 0, 0)
                    pdf.multi_cell(0, 4.5, content)
                    pdf.ln(1.5)
            
            elif elem_type == 'li':
                content = safe_text(content)
                pdf.set_font(main_font, '', 9)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(8, 4.5, '  *')
                pdf.multi_cell(0, 4.5, content)
            
            elif elem_type == 'code':
                pdf.ln(1)
                pdf.set_fill_color(45, 55, 72)
                pdf.set_text_color(226, 232, 240)
                if 'DejaVuMono' in [f for f in pdf.fonts]:
                    pdf.set_font('DejaVuMono', '', 7.5)
                else:
                    pdf.set_font('Courier', '', 7.5)
                lines = content.split('\n')
                for line in lines:
                    line = safe_text(line)
                    pdf.cell(0, 3.5, '  ' + line, 0, 1, fill=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(1)
            
            elif elem_type == 'hr':
                pdf.ln(4)
                pdf.set_draw_color(203, 213, 224)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(4)
            
            elif elem_type == 'table':
                pdf.ln(2)
                table_lines = content
                rows = []
                for tl in table_lines:
                    if '---' not in tl:
                        cells = [c.strip() for c in tl.split('|')[1:-1]]
                        rows.append(cells)
                
                if rows:
                    num_cols = len(rows[0])
                    col_width = min(170 / num_cols, 60)
                    
                    # Header
                    pdf.set_font(main_font, 'B', 8)
                    pdf.set_fill_color(30, 58, 95)
                    pdf.set_text_color(255, 255, 255)
                    for cell in rows[0]:
                        cell = safe_text(cell, 28)
                        pdf.cell(col_width, 6, cell, 1, 0, 'L', fill=True)
                    pdf.ln()
                    
                    # Data rows
                    pdf.set_font(main_font, '', 8)
                    pdf.set_text_color(0, 0, 0)
                    for i, row in enumerate(rows[1:]):
                        if i % 2 == 0:
                            pdf.set_fill_color(247, 250, 252)
                        else:
                            pdf.set_fill_color(255, 255, 255)
                        for cell in row:
                            cell = safe_text(cell, 32)
                            pdf.cell(col_width, 5, cell, 1, 0, 'L', fill=True)
                        pdf.ln()
                pdf.ln(2)
            
            elif elem_type == 'space':
                pdf.ln(1.5)
        except Exception as e:
            print(f"Warning: Error processing element {elem_type}: {e}")
            continue
    
    # Save PDF
    pdf_path = BASE_DIR / "Gene-Brain_CCA_Concise_Bilingual_Report.pdf"
    pdf.output(str(pdf_path), 'F')
    print(f"  Concise bilingual PDF saved to: {pdf_path}")
    return pdf_path


def main():
    """Generate all PDF reports."""
    print("="*60)
    print("Gene-Brain CCA Report PDF Generator")
    print("="*60)
    print()
    
    # Create output directory if needed
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate comprehensive PDF
    comprehensive_pdf = render_comprehensive_pdf()
    
    # Generate concise bilingual PDF
    concise_pdf = render_concise_pdf()
    
    print()
    print("="*60)
    print("PDF Generation Complete!")
    print("="*60)
    print()
    print("Generated files:")
    print(f"  1. {comprehensive_pdf}")
    print(f"  2. {concise_pdf}")
    print()


if __name__ == "__main__":
    main()
