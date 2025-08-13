"""PDF report generation for SurgicalAI."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from .schemas import ReportRequest


logger = logging.getLogger(__name__)


async def generate_report_pdf(request: ReportRequest, request_id: str) -> bytes:
    """Generate a professional PDF report from analysis results and return PDF bytes."""
    logger.info(f"Generating PDF report for request {request_id}")

    try:
        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build content
        story: list = []
        styles = get_custom_styles()

        # Sections
        story.extend(build_header(styles, request_id))
        story.extend(build_analysis_info(request.analysis_payload, styles))
        story.extend(build_diagnosis_section(request.analysis_payload, styles))
        story.extend(build_oncology_section(request.analysis_payload, styles))
        story.extend(build_reconstruction_section(request.analysis_payload, styles))
        story.extend(build_protocol_section(request.analysis_payload, styles))
        story.extend(build_images_section(request.analysis_payload, styles))
        story.extend(build_citations_section(request.analysis_payload, styles))
        story.extend(build_signature_section(request.doctor_name, styles))

        # Build PDF
        doc.build(story)

        # Return bytes
        pdf_content = buffer.getvalue()
        buffer.close()
        logger.info(f"PDF report generated ({len(pdf_content)} bytes)")
        return pdf_content
    except Exception as e:
        logger.error(f"PDF generation failed for request {request_id}: {e}", exc_info=True)
        raise


def get_custom_styles():
    """Get custom styles for the PDF."""
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2563eb')
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#1f2937')
    ))
    
    styles.add(ParagraphStyle(
        name='Subsection',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.HexColor('#374151')
    ))
    
    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey
    ))
    
    return styles


def build_header(styles, request_id: str):
    """Build report header."""
    elements = []
    
    # Title
    title = Paragraph("SurgicalAI – Tension Field + Parametric Flap Solver", styles['CustomTitle'])
    elements.append(title)
    
    # Subtitle
    subtitle = Paragraph("Comprehensive Dermatologic Analysis Report", styles['Normal'])
    subtitle.style.alignment = TA_CENTER
    subtitle.style.fontSize = 12
    subtitle.style.textColor = colors.grey
    elements.append(subtitle)
    
    elements.append(Spacer(1, 20))
    
    # Report info table
    report_data = [
        ['Report ID:', request_id],
        ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M')],
        ['Analysis Type:', 'AI-Powered Lesion Assessment']
    ]
    
    info_table = Table(report_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    elements.append(info_table)
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.grey))
    
    return elements


def build_citations_section(analysis_data: Dict[str, Any], styles):
    """Build citations/references section if available."""
    elements = []
    
    citations = analysis_data.get('citations')
    if not citations:
        # Try nested payloads
        citations = (
            analysis_data.get('analysis', {}).get('citations') or
            analysis_data.get('diagnostics', {}).get('citations') or
            []
        )
    
    if citations:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("References", styles['SectionHeader']))
        
        # Render as numbered list
        for idx, cite in enumerate(citations, start=1):
            elements.append(Paragraph(f"{idx}. {cite}", styles['Normal']))
    
    return elements


def build_analysis_info(analysis_data: Dict[str, Any], styles):
    """Build analysis information section."""
    elements = []
    
    elements.append(Paragraph("Analysis Parameters", styles['SectionHeader']))
    
    # Extract analysis parameters
    params_data = []
    if 'site' in analysis_data or analysis_data.get('request', {}).get('site'):
        site = analysis_data.get('site') or analysis_data.get('request', {}).get('site')
        params_data.append(['Anatomical Site:', site.replace('_', ' ').title()])
    
    if 'rstl_angle_deg' in analysis_data:
        params_data.append(['RSTL Angle:', f"{analysis_data['rstl_angle_deg']:.1f}°"])
    
    if 'tension_score' in analysis_data:
        params_data.append(['Tension Score:', f"{analysis_data['tension_score']:.2f}"])
    
    if params_data:
        params_table = Table(params_data, colWidths=[2*inch, 3*inch])
        params_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(params_table)
    
    return elements


def build_diagnosis_section(analysis_data: Dict[str, Any], styles):
    """Build diagnosis section."""
    elements = []
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Diagnostic Analysis", styles['SectionHeader']))
    
    # Get diagnostics
    diagnostics = analysis_data.get('diagnostics', {})
    top3 = diagnostics.get('top3', [])
    
    if top3:
        elements.append(Paragraph("Differential Diagnosis (Top 3)", styles['Subsection']))
        
        # Create diagnosis table
        diag_data = [['Diagnosis', 'Probability', 'Confidence']]
        
        for i, diag in enumerate(top3[:3]):
            label = diag.get('label', '').replace('_', ' ').title()
            prob = diag.get('prob', 0)
            confidence = 'High' if prob > 0.7 else 'Moderate' if prob > 0.4 else 'Low'
            diag_data.append([label, f"{prob:.1%}", confidence])
        
        diag_table = Table(diag_data, colWidths=[2.5*inch, 1*inch, 1*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(diag_table)
    
    # Clinical notes
    notes = diagnostics.get('notes', '')
    if notes:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Clinical Notes", styles['Subsection']))
        elements.append(Paragraph(notes, styles['Normal']))
    
    # ABCDE analysis if available
    abcde = analysis_data.get('abcde', {})
    if abcde:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("ABCDE Analysis", styles['Subsection']))
        
        abcde_data = [
            ['Asymmetry', f"{abcde.get('asymmetry', 0):.2f}"],
            ['Border Irregularity', f"{abcde.get('border', 0):.2f}"],
            ['Color Variation', f"{abcde.get('color', 0):.2f}"],
            ['Diameter', f"{abcde.get('diameter', 0):.2f}"],
            ['Evolving', f"{abcde.get('evolving', 0):.2f}"],
            ['Overall Score', f"{abcde.get('overall_score', 0):.2f}"]
        ]
        
        abcde_table = Table(abcde_data, colWidths=[2*inch, 1*inch])
        abcde_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(abcde_table)
    
    return elements


def build_reconstruction_section(analysis_data: Dict[str, Any], styles):
    """Build reconstruction planning section."""
    elements = []
    
    # Accept either classic 'flap_plan' or triage 'reconstruction'
    flap_plan = analysis_data.get('flap_plan') or analysis_data.get('reconstruction') or {}
    if not flap_plan:
        return elements
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Surgical Reconstruction Plan", styles['SectionHeader']))
    
    # Primary flap recommendation
    flap_type = flap_plan.get('type', '').replace('_', ' ').title() or (flap_plan.get('primary', {}).get('type', '').replace('_',' ').title())
    elements.append(Paragraph(f"Recommended Technique: {flap_type} Flap", styles['Subsection']))
    
    # Flap parameters
    params = flap_plan.get('params') or flap_plan.get('primary', {}).get('params', {})
    if params:
        param_data = []
        
        if 'defect_d_mm' in params:
            param_data.append(['Defect Diameter:', f"{params['defect_d_mm']:.1f} mm"])
        if 'margin_mm' in params:
            param_data.append(['Surgical Margin:', f"{params['margin_mm']:.1f} mm"])
        if 'arc_deg' in params:
            param_data.append(['Flap Arc:', f"{params['arc_deg']:.0f}°"])
        if 'base_width_mm' in params:
            param_data.append(['Base Width:', f"{params['base_width_mm']:.1f} mm"])
        if 'angle_to_rstl_deg' in params:
            param_data.append(['Angle to RSTL:', f"{params['angle_to_rstl_deg']:.1f}°"])
        
        if param_data:
            param_table = Table(param_data, colWidths=[2*inch, 1.5*inch])
            param_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(param_table)
    
    # Alternative options
    alternatives = flap_plan.get('alternatives', []) or flap_plan.get('alternatives', [])
    if alternatives:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Alternative Options", styles['Subsection']))
        
        alt_data = [['Technique', 'Suitability Score']]
        for alt in alternatives:
            alt_type = alt.get('type', '').replace('_', ' ').title()
            score = alt.get('score', 0)
            alt_data.append([f"{alt_type} Flap", f"{score:.1%}"])
        
        alt_table = Table(alt_data, colWidths=[2*inch, 1.5*inch])
        alt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(alt_table)
    
    return elements


def build_oncology_section(analysis_data: Dict[str, Any], styles):
    """Build oncologic recommendation section (margins, gates)."""
    elements = []
    oncology = analysis_data.get('oncology') or analysis_data.get('plan', {}).get('oncology')
    if not oncology:
        return elements

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Oncologic Planning", styles['SectionHeader']))

    data = []
    if 'tumor_size_mm' in oncology:
        data.append(['Estimated Tumor Size:', f"{oncology['tumor_size_mm']:.1f} mm"])
    if 'recommended_margin_mm' in oncology:
        data.append(['Recommended Margin:', f"{oncology['recommended_margin_mm']:.1f} mm"])
    if 'technique' in oncology:
        data.append(['Technique:', str(oncology['technique'])])

    gates = oncology.get('gates') or {}
    if gates:
        gate_str = ', '.join([f"{k.replace('_',' ').title()}: {'Yes' if v else 'No'}" for k, v in gates.items()])
        data.append(['Safety Gates:', gate_str])

    if data:
        tbl = Table(data, colWidths=[2.2*inch, 3.3*inch])
        tbl.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(tbl)

    return elements


def build_protocol_section(analysis_data: Dict[str, Any], styles):
    """Build protocol/guideline section summarizing applied rules."""
    elements = []
    protocol = analysis_data.get('protocol')
    if not protocol:
        return elements

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Protocols & Guidelines Applied", styles['SectionHeader']))

    name = protocol.get('name', 'SurgicalAI Clinical Protocol')
    version = protocol.get('version', '')
    sources = protocol.get('sources', [])
    decisions = protocol.get('decisions', {})

    header = [["Protocol:", f"{name} v{version}"]]
    tbl = Table(header, colWidths=[1.8*inch, 3.7*inch])
    tbl.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    elements.append(tbl)

    if sources:
        elements.append(Paragraph("Sources:", styles['Subsection']))
        for s in sources:
            elements.append(Paragraph(f"• {s}", styles['Normal']))

    if decisions:
        elements.append(Paragraph("Key Decisions:", styles['Subsection']))
        for k, v in decisions.items():
            val = v
            if isinstance(v, bool):
                val = 'Yes' if v else 'No'
            elif isinstance(v, float):
                val = f"{v:.2f}"
            elements.append(Paragraph(f"• {k.replace('_',' ').title()}: {val}", styles['Normal']))

    return elements


def build_images_section(analysis_data: Dict[str, Any], styles):
    """Build images section with artifacts."""
    elements = []
    
    artifacts = analysis_data.get('artifacts', {})
    if not artifacts:
        return elements
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Visual Analysis", styles['SectionHeader']))
    
    try:
        # Add overlay image if available
        overlay_b64 = artifacts.get('overlay_png_base64')
        if overlay_b64:
            elements.append(Paragraph("ROI Overlay", styles['Subsection']))
            
            # Decode base64 image
            import base64
            overlay_data = base64.b64decode(overlay_b64)
            overlay_buffer = io.BytesIO(overlay_data)
            
            # Add to PDF with max width
            overlay_img = RLImage(overlay_buffer, width=4*inch, height=3*inch)
            elements.append(overlay_img)
            elements.append(Spacer(1, 10))
        
        # Add heatmap if available  
        heatmap_b64 = artifacts.get('heatmap_png_base64')
        if heatmap_b64:
            elements.append(Paragraph("Attention Heatmap", styles['Subsection']))
            
            heatmap_data = base64.b64decode(heatmap_b64)
            heatmap_buffer = io.BytesIO(heatmap_data)
            
            heatmap_img = RLImage(heatmap_buffer, width=4*inch, height=3*inch)
            elements.append(heatmap_img)
            
    except Exception as e:
        logger.warning(f"Failed to add images to PDF: {e}")
        elements.append(Paragraph("Image artifacts could not be embedded.", styles['Normal']))
    
    return elements


def build_signature_section(doctor_name: str, styles):
    """Build signature section."""
    elements = []
    
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.grey))
    elements.append(Spacer(1, 15))
    
    # Disclaimer
    disclaimer = Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by AI for educational and research purposes only. "
        "All clinical decisions must be made by qualified medical professionals. This analysis does not replace "
        "clinical examination, histopathological diagnosis, or professional medical judgment.",
        styles['Normal']
    )
    disclaimer.style.fontSize = 9
    disclaimer.style.textColor = colors.red
    elements.append(disclaimer)
    
    elements.append(Spacer(1, 20))
    
    # Signature
    signature = Paragraph(
        f"<b>{doctor_name}</b><br/>"
        "Plastic, Reconstructive &amp; Aesthetic Surgery<br/>"
        f"Report generated: {datetime.now().strftime('%B %d, %Y')}",
        styles['Footer']
    )
    signature.style.alignment = TA_RIGHT
    elements.append(signature)
    
    return elements
